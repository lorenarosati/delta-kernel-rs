//! Shared parquet comparison utilities for integration tests.
//!
//! Extracted from golden_tables.rs so that both golden table tests and validation tests
//! can reuse the same comparison logic.

use std::path::Path;
use std::sync::Arc;

use delta_kernel::arrow::array::{Array, AsArray, StructArray};
use delta_kernel::arrow::compute::{concat_batches, lexsort_to_indices, take, SortColumn};
use delta_kernel::arrow::datatypes::{DataType, FieldRef};
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::parquet::arrow::async_reader::{
    ParquetObjectReader, ParquetRecordBatchStreamBuilder,
};
use delta_kernel::DeltaResult;

use futures::{stream::TryStreamExt, StreamExt};
use object_store::local::LocalFileSystem;
use object_store::ObjectStore;

/// Read all parquet files in the directory and concatenate them into a single RecordBatch.
/// Adapted from DAT (Delta Acceptance Tests).
pub async fn read_expected(path: &Path) -> DeltaResult<RecordBatch> {
    let store = Arc::new(LocalFileSystem::new_with_prefix(path)?);
    let files = store.list(None).try_collect::<Vec<_>>().await?;
    let mut batches = vec![];
    let mut schema = None;
    for meta in files.into_iter() {
        if let Some(ext) = meta.location.extension() {
            if ext == "parquet" {
                let reader = ParquetObjectReader::new(store.clone(), meta.location);
                let builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
                if schema.is_none() {
                    schema = Some(builder.schema().clone());
                }
                let mut stream = builder.build()?;
                while let Some(batch) = stream.next().await {
                    batches.push(batch?);
                }
            }
        }
    }
    let all_data = concat_batches(&schema.unwrap(), &batches)?;
    Ok(all_data)
}

/// Sort a RecordBatch by all sortable columns for deterministic comparison.
/// Copied from DAT.
pub fn sort_record_batch(batch: RecordBatch) -> DeltaResult<RecordBatch> {
    if batch.num_rows() < 2 {
        return Ok(batch);
    }
    let mut sort_columns = vec![];
    for col in batch.columns() {
        match col.data_type() {
            DataType::Struct(_) | DataType::Map(_, _) => {
                // can't sort by structs or maps
            }
            DataType::List(list_field) => {
                let list_dt = list_field.data_type();
                if list_dt.is_primitive() {
                    sort_columns.push(SortColumn {
                        values: col.clone(),
                        options: None,
                    })
                }
            }
            _ => sort_columns.push(SortColumn {
                values: col.clone(),
                options: None,
            }),
        }
    }
    let indices = lexsort_to_indices(&sort_columns, None)?;
    let columns = batch
        .columns()
        .iter()
        .map(|c| take(c, &indices, None).unwrap())
        .collect();
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

/// Ensure that two sets of fields have the same names and dict_is_ordered.
/// We ignore data type (checked in assert_columns_match), nullability (parquet marks many things
/// as nullable), and metadata (diverges between real and golden data).
pub fn assert_fields_match<'a>(
    actual: impl Iterator<Item = &'a FieldRef>,
    expected: impl Iterator<Item = &'a FieldRef>,
) {
    for (actual_field, expected_field) in actual.zip(expected) {
        assert!(
            actual_field.name() == expected_field.name(),
            "Field names don't match"
        );
        assert!(
            actual_field.dict_is_ordered() == expected_field.dict_is_ordered(),
            "Field dict_is_ordered doesn't match"
        );
    }
}

/// Recursively compare two arrow arrays, handling nested types (struct, list, map).
pub fn assert_cols_eq(actual: &dyn Array, expected: &dyn Array) {
    match actual.data_type() {
        DataType::Struct(_) => {
            let actual_sa = actual.as_struct();
            let expected_sa = expected.as_struct();
            assert_struct_eq(actual_sa, expected_sa);
        }
        DataType::List(_) => {
            let actual_la = actual.as_list::<i32>();
            let expected_la = expected.as_list::<i32>();
            assert_cols_eq(actual_la.values(), expected_la.values());
        }
        DataType::Map(_, _) => {
            let actual_ma = actual.as_map();
            let expected_ma = expected.as_map();
            assert_cols_eq(actual_ma.keys(), expected_ma.keys());
            assert_cols_eq(actual_ma.values(), expected_ma.values());
        }
        _ => {
            assert_eq!(actual, expected, "Column data didn't match.");
        }
    }
}

/// Compare two StructArrays field-by-field.
pub fn assert_struct_eq(actual: &StructArray, expected: &StructArray) {
    let actual_fields = actual.fields();
    let expected_fields = expected.fields();
    assert_eq!(
        actual_fields.len(),
        expected_fields.len(),
        "Number of fields differed"
    );
    assert_fields_match(actual_fields.iter(), expected_fields.iter());
    let actual_cols = actual.columns();
    let expected_cols = expected.columns();
    assert_eq!(
        actual_cols.len(),
        expected_cols.len(),
        "Number of columns differed"
    );
    for (actual_col, expected_col) in actual_cols.iter().zip(expected_cols) {
        assert_cols_eq(actual_col, expected_col);
    }
}


