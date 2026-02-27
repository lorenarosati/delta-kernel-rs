//! Validation tests for read specs.
//!
//! Loads read specs from `kernel_benchmark_specs.tar.gz`, runs ReadData and ReadMetadata
//! operations, and compares results against expected parquet files.
//!
//! Only read specs without predicates are tested (snapshot specs and predicate specs are skipped).

use std::io::{BufReader, Write as _};
use std::panic;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use delta_kernel::arrow::array::{Array as _, AsArray as _};
use delta_kernel::arrow::compute::concat_batches;
use delta_kernel::arrow::datatypes::Schema as ArrowSchema;
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::benchmarks::models::{ReadSpec, TableInfo};
use delta_kernel::engine::arrow_conversion::TryFromKernel as _;
use delta_kernel::engine::arrow_data::EngineDataArrowExt;
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::scan::state::ScanFile;
use delta_kernel::Snapshot;

use flate2::read::GzDecoder;
use object_store::local::LocalFileSystem;
use serde::Deserialize;
use tar::Archive;

mod common;
use common::comparison::{assert_struct_eq, read_expected, sort_record_batch};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const VALIDATION_TAR: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/kernel_benchmark_specs.tar (3).gz");
const OUTPUT_FOLDER: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/validation_specs");
const DONE_FILE: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/validation_specs/.done");
const TABLE_INFO_FILE: &str = "table_info.json";
const SPECS_DIR: &str = "specs";
const EXPECTED_DIR: &str = "expected";

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// A single validation case: one read spec for one table, with paths to expected results.
struct ValidationCase {
    table_info: TableInfo,
    case_name: String,
    read_spec: ReadSpec,
    expected_dir: PathBuf, // path to expected/<spec_name>/
}

/// Deserialized summary.json from expected results.
#[derive(Debug, Deserialize)]
struct ExpectedSummary {
    expected_row_count: usize,
    file_count: usize,
    #[allow(dead_code)]
    total_file_count: usize,
    #[allow(dead_code)]
    files_skipped: usize,
    #[allow(dead_code)]
    actual_row_count: usize,
    #[allow(dead_code)]
    matches_expected: bool,
}

// ---------------------------------------------------------------------------
// Loader: extract tar.gz, walk tables, parse/filter specs
// ---------------------------------------------------------------------------

fn ensure_extracted() -> Result<(), Box<dyn std::error::Error>> {
    if Path::new(DONE_FILE).exists() {
        return Ok(());
    }

    let tar_path = Path::new(VALIDATION_TAR);
    if !tar_path.exists() {
        return Err(format!("Validation tarball not found at {}", VALIDATION_TAR).into());
    }

    let file = std::fs::File::open(tar_path)?;
    let tarball = GzDecoder::new(BufReader::new(file));
    let mut archive = Archive::new(tarball);

    std::fs::create_dir_all(OUTPUT_FOLDER)?;
    archive.unpack(OUTPUT_FOLDER)?;

    let mut done = std::fs::File::create(DONE_FILE)?;
    write!(done, "done")?;

    Ok(())
}

fn load_validation_cases() -> Result<Vec<ValidationCase>, Box<dyn std::error::Error>> {
    ensure_extracted()?;

    let base_dir = PathBuf::from(OUTPUT_FOLDER);
    let table_dirs = find_table_directories(&base_dir)?;

    let mut cases = Vec::new();
    for table_dir in &table_dirs {
        if let Ok(table_cases) = load_cases_from_table(table_dir) {
            cases.extend(table_cases);
        }
    }

    Ok(cases)
}

fn find_table_directories(base_dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let entries = std::fs::read_dir(base_dir)?;
    let mut table_dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.file_name().map_or(false, |n| n != ".done"))
        .collect();
    table_dirs.sort();
    Ok(table_dirs)
}

fn load_cases_from_table(
    table_dir: &Path,
) -> Result<Vec<ValidationCase>, Box<dyn std::error::Error>> {
    let specs_dir = table_dir.join(SPECS_DIR);
    if !specs_dir.is_dir() {
        return Err(format!("No specs dir: {}", specs_dir.display()).into());
    }

    let table_info_path = table_dir.join(TABLE_INFO_FILE);
    let table_info = TableInfo::from_json_path(&table_info_path)?;

    // Verify delta/ exists for tables without explicit table_path
    if table_info.table_path.is_none() {
        let delta_dir = table_dir.join("delta");
        if !delta_dir.is_dir() {
            return Err(format!(
                "No delta/ dir for table '{}' at {}",
                table_info.name,
                table_dir.display()
            )
            .into());
        }
    }

    let expected_base = table_dir.join(EXPECTED_DIR);
    let spec_files = find_spec_files(&specs_dir)?;

    let mut cases = Vec::new();
    for spec_file in &spec_files {
        if let Some(case) = try_load_read_case(spec_file, &table_info, &expected_base)? {
            cases.push(case);
        }
    }

    Ok(cases)
}

fn find_spec_files(specs_dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let entries = std::fs::read_dir(specs_dir)?;
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().map_or(false, |ext| ext == "json"))
        .collect();
    files.sort();
    Ok(files)
}

/// Attempt to load a spec file as a read-without-predicate validation case.
/// Returns None if the spec is not a read spec, has a predicate, or has no expected dir.
fn try_load_read_case(
    spec_file: &Path,
    table_info: &TableInfo,
    expected_base: &Path,
) -> Result<Option<ValidationCase>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(spec_file)?;
    let raw: serde_json::Value = serde_json::from_str(&content)?;

    // Only process read specs
    let spec_type = raw.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if spec_type != "read" {
        return Ok(None);
    }

    // Skip specs with predicates
    if raw.get("predicate").is_some() {
        return Ok(None);
    }

    // Parse as ReadSpec
    let read_spec: ReadSpec = serde_json::from_value(
        // Remove the "type" field since ReadSpec doesn't have it
        // ReadSpec only has { version: Option<u64> }
        {
            let mut map = raw.as_object().cloned().unwrap_or_default();
            map.remove("type");
            serde_json::Value::Object(map)
        },
    )?;

    let case_name = spec_file
        .file_stem()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("Invalid spec file name: {}", spec_file.display()))?
        .to_string();

    let expected_dir = expected_base.join(&case_name);
    if !expected_dir.is_dir() {
        return Ok(None);
    }

    Ok(Some(ValidationCase {
        table_info: table_info.clone(),
        case_name,
        read_spec,
        expected_dir,
    }))
}

fn load_summary(expected_dir: &Path) -> Result<ExpectedSummary, Box<dyn std::error::Error>> {
    let summary_path = expected_dir.join("summary.json");
    let content = std::fs::read_to_string(&summary_path)?;
    let summary: ExpectedSummary = serde_json::from_str(&content)?;
    Ok(summary)
}

// ---------------------------------------------------------------------------
// Engine setup
// ---------------------------------------------------------------------------

fn setup_engine() -> Arc<DefaultEngine<TokioBackgroundExecutor>> {
    let store = Arc::new(LocalFileSystem::new());
    let engine = DefaultEngine::builder(store).build();
    Arc::new(engine)
}

fn build_snapshot(
    table_info: &TableInfo,
    read_spec: &ReadSpec,
    engine: &DefaultEngine<TokioBackgroundExecutor>,
) -> Result<Arc<Snapshot>, Box<dyn std::error::Error>> {
    let table_root = table_info.resolved_table_root();
    let url = delta_kernel::try_parse_uri(table_root)?;
    let mut builder = Snapshot::builder_for(url);
    if let Some(version) = read_spec.version {
        builder = builder.at_version(version);
    }
    Ok(builder.build(engine)?)
}

// ---------------------------------------------------------------------------
// ReadData validation
// ---------------------------------------------------------------------------

/// Validates ReadData and returns the actual row count on success.
async fn validate_read_data(
    case: &ValidationCase,
    engine: &Arc<DefaultEngine<TokioBackgroundExecutor>>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let snapshot = build_snapshot(&case.table_info, &case.read_spec, engine)?;
    let scan = snapshot.scan_builder().build()?;
    let scan_res = scan.execute(engine.clone())?;

    let batches: Vec<RecordBatch> = scan_res
        .map(EngineDataArrowExt::try_into_record_batch)
        .collect::<Result<Vec<_>, _>>()?;

    let expected_data_dir = case.expected_dir.join("expected_data");
    if !expected_data_dir.is_dir() {
        return Err(format!(
            "No expected_data dir: {}",
            expected_data_dir.display()
        )
        .into());
    }

    let expected = read_expected(&expected_data_dir).await?;

    let schema = Arc::new(ArrowSchema::try_from_kernel(scan.logical_schema().as_ref())?);
    let result = if batches.is_empty() {
        RecordBatch::new_empty(schema)
    } else {
        concat_batches(&schema, &batches)?
    };

    let result = sort_record_batch(result)?;
    let expected = sort_record_batch(expected)?;

    if result.num_rows() != expected.num_rows() {
        return Err(format!(
            "Row count mismatch: got {} rows, expected {}",
            result.num_rows(),
            expected.num_rows()
        )
        .into());
    }

    let row_count = result.num_rows();

    let result_struct = result.into();
    let expected_struct = expected.into();
    // Use catch_unwind so assertion panics become errors (allowing us to collect all failures)
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        assert_struct_eq(&result_struct, &expected_struct);
    }))
    .map_err(|e| {
        let msg = e
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| e.downcast_ref::<&str>().copied())
            .unwrap_or("comparison panicked");
        format!("Data comparison failed: {}", msg)
    })?;

    Ok(row_count)
}

// ---------------------------------------------------------------------------
// ReadMetadata validation
// ---------------------------------------------------------------------------

/// Validates ReadMetadata and returns the actual scan file count on success.
async fn validate_read_metadata(
    case: &ValidationCase,
    engine: &Arc<DefaultEngine<TokioBackgroundExecutor>>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let snapshot = build_snapshot(&case.table_info, &case.read_spec, engine)?;
    let scan = snapshot.scan_builder().build()?;
    let scan_metadata_iter = scan.scan_metadata(engine.as_ref())?;

    // Collect all scan file paths
    fn scan_file_callback(files: &mut Vec<ScanFile>, file: ScanFile) {
        files.push(file);
    }

    let mut all_scan_files: Vec<ScanFile> = Vec::new();
    for result in scan_metadata_iter {
        let scan_metadata = result?;
        all_scan_files = scan_metadata.visit_scan_files(all_scan_files, scan_file_callback)?;
    }

    let expected_metadata_dir = case.expected_dir.join("expected_metadata");
    if !expected_metadata_dir.is_dir() {
        return Err(format!(
            "No expected_metadata dir: {}",
            expected_metadata_dir.display()
        )
        .into());
    }

    // Read expected metadata parquet -- it has an `action` column with JSON add-action strings
    let expected_batch = read_expected(&expected_metadata_dir).await?;

    // Extract file paths from expected add actions
    let action_col = expected_batch
        .column_by_name("action")
        .ok_or("expected_metadata parquet missing 'action' column")?;
    let action_strings = action_col.as_string::<i32>();

    let mut expected_paths: Vec<String> = Vec::new();
    for i in 0..action_strings.len() {
        let action_json = action_strings.value(i);
        let parsed: serde_json::Value = serde_json::from_str(action_json)?;
        if let Some(path) = parsed
            .get("add")
            .and_then(|a| a.get("path"))
            .and_then(|p| p.as_str())
        {
            expected_paths.push(path.to_string());
        }
    }

    let mut actual_paths: Vec<String> = all_scan_files.iter().map(|f| f.path.clone()).collect();
    actual_paths.sort();
    expected_paths.sort();

    if actual_paths.len() != expected_paths.len() {
        return Err(format!(
            "Metadata file count mismatch: got {} files, expected {}",
            actual_paths.len(),
            expected_paths.len()
        )
        .into());
    }

    if actual_paths != expected_paths {
        return Err(format!(
            "Metadata file paths mismatch.\nActual:   {:?}\nExpected: {:?}",
            actual_paths, expected_paths
        )
        .into());
    }

    Ok(actual_paths.len())
}

// ---------------------------------------------------------------------------
// Summary.json sanity checks
// ---------------------------------------------------------------------------

fn validate_summary(
    case: &ValidationCase,
    actual_data_row_count: usize,
    actual_file_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = load_summary(&case.expected_dir)?;

    if actual_data_row_count != summary.expected_row_count {
        return Err(format!(
            "Summary row count mismatch: got {}, expected {}",
            actual_data_row_count, summary.expected_row_count
        )
        .into());
    }

    if actual_file_count != summary.file_count {
        return Err(format!(
            "Summary file count mismatch: got {}, expected {}",
            actual_file_count, summary.file_count
        )
        .into());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn validate_read_specs() {
    let cases = load_validation_cases().expect("Failed to load validation cases");
    assert!(!cases.is_empty(), "No validation cases found");

    let engine = setup_engine();
    let mut failures: Vec<String> = Vec::new();

    for case in &cases {
        let label = format!("{}/{}", case.table_info.name, case.case_name);

        // -- ReadData validation --
        let data_row_count = match validate_read_data(case, &engine).await {
            Ok(row_count) => Some(row_count),
            Err(e) => {
                failures.push(format!("{}: ReadData failed: {}", label, e));
                None
            }
        };

        // -- ReadMetadata validation --
        let file_count = match validate_read_metadata(case, &engine).await {
            Ok(count) => Some(count),
            Err(e) => {
                failures.push(format!("{}: ReadMetadata failed: {}", label, e));
                None
            }
        };

        // -- Summary check --
        if let (Some(rows), Some(files)) = (data_row_count, file_count) {
            if let Err(e) = validate_summary(case, rows, files) {
                failures.push(format!("{}: Summary check failed: {}", label, e));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "\n{} validation failure(s) out of {} cases:\n{}",
            failures.len(),
            cases.len(),
            failures.join("\n")
        );
    }

    println!(
        "All {} validation cases passed (ReadData + ReadMetadata + Summary)",
        cases.len()
    );
}

