//! Benchmark runners for executing Delta table operations.
//!
//! Each runner handles setup and execution of a specific operation (e.g., reading metadata)
//! Results are discarded for benchmarking purposes

use crate::actions::deletion_vector::split_vector;
use crate::benchmarks::models::{
    ParallelScan, ReadConfig, ReadOperation, ReadSpec, SnapshotConstructionSpec, TableInfo,
};
use crate::parallel::parallel_phase::ParallelPhase;
use crate::parallel::sequential_phase::AfterSequential;
use crate::scan::state;
use crate::scan::ScanMetadata;
use crate::schema::SchemaRef;
use crate::snapshot::Snapshot;
use crate::{Engine, FileMeta};

use std::hint::black_box;
use std::sync::Arc;
use std::thread;
use url::Url;

/// Each runner holds all the state required for its workload (e.g. read metadata needs pre-built snapshots and a config)
/// so that `execute` measures only the operation itself
pub trait WorkloadRunner {
    fn execute(&self) -> Result<(), Box<dyn std::error::Error>>;
    fn name(&self) -> &str;
}

/// Shared setup state for read runners (both ReadMetadata and ReadData).
struct ReadSetup {
    snapshot: Arc<Snapshot>,
    engine: Arc<dyn Engine>,
    name: String,
}

/// Builds the snapshot and benchmark name that both read runners need.
fn build_read_setup(
    table_info: &TableInfo,
    case_name: &str,
    read_spec: &ReadSpec,
    operation: ReadOperation,
    config: &ReadConfig,
    engine: Arc<dyn Engine>,
) -> Result<ReadSetup, Box<dyn std::error::Error>> {
    let table_root = table_info.resolved_table_root();
    let url = crate::try_parse_uri(table_root)?;

    let mut builder = Snapshot::builder_for(url);
    if let Some(version) = read_spec.version {
        builder = builder.at_version(version);
    }

    let snapshot = builder.build(engine.as_ref())?;

    let name = format!(
        "{}/{}/{}/{}",
        table_info.name,
        case_name,
        operation.as_str(),
        config.name,
    );

    Ok(ReadSetup {
        snapshot,
        engine,
        name,
    })
}

pub struct ReadMetadataRunner {
    snapshot: Arc<Snapshot>,
    engine: Arc<dyn Engine>,
    name: String,
    config: ReadConfig,
}

impl ReadMetadataRunner {
    pub fn setup(
        table_info: &TableInfo,
        case_name: &str,
        read_spec: &ReadSpec,
        config: ReadConfig,
        engine: Arc<dyn Engine>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let s = build_read_setup(
            table_info,
            case_name,
            read_spec,
            ReadOperation::ReadMetadata,
            &config,
            engine,
        )?;
        Ok(Self {
            snapshot: s.snapshot,
            engine: s.engine,
            name: s.name,
            config,
        })
    }

    fn execute_serial(&self) -> Result<(), Box<dyn std::error::Error>> {
        let scan = self.snapshot.clone().scan_builder().build()?;
        let metadata_iter = scan.scan_metadata(self.engine.as_ref())?;
        for result in metadata_iter {
            black_box(result?);
        }
        Ok(())
    }

    fn execute_parallel(&self, num_threads: usize) -> Result<(), Box<dyn std::error::Error>> {
        let scan = self.snapshot.clone().scan_builder().build()?;

        let mut phase1 = scan.parallel_scan_metadata(self.engine.clone())?;
        for result in phase1.by_ref() {
            black_box(result?);
        }

        match phase1.finish()? {
            AfterSequential::Done(_) => {}
            AfterSequential::Parallel { processor, files } => {
                if num_threads == 0 {
                    return Err("num_threads in ReadConfig must be greater than 0".into());
                }
                let files_per_worker = files.len().div_ceil(num_threads);

                let partitions: Vec<_> = files
                    .chunks(files_per_worker)
                    .map(|chunk| chunk.to_vec())
                    .collect();

                let processor = Arc::new(processor);

                let handles: Vec<_> = partitions
                    .into_iter()
                    .map(|partition_files| {
                        let engine = self.engine.clone();
                        let processor = processor.clone();

                        thread::spawn(move || -> Result<(), crate::Error> {
                            if partition_files.is_empty() {
                                return Ok(());
                            }

                            let parallel =
                                ParallelPhase::try_new(engine, processor, partition_files)?;
                            for result in parallel {
                                black_box(result?);
                            }

                            Ok(())
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().map_err(|_| -> Box<dyn std::error::Error> {
                        "Worker thread panicked".into()
                    })??;
                }
            }
        }
        Ok(())
    }
}

impl WorkloadRunner for ReadMetadataRunner {
    fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        match &self.config.parallel_scan {
            ParallelScan::Disabled => {
                self.execute_serial()?;
            }
            ParallelScan::Enabled { num_threads } => {
                self.execute_parallel(*num_threads)?;
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub struct ReadDataRunner {
    snapshot: Arc<Snapshot>,
    engine: Arc<dyn Engine>,
    name: String,
    config: ReadConfig,
}

impl ReadDataRunner {
    pub fn setup(
        table_info: &TableInfo,
        case_name: &str,
        read_spec: &ReadSpec,
        config: ReadConfig,
        engine: Arc<dyn Engine>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let s = build_read_setup(
            table_info,
            case_name,
            read_spec,
            ReadOperation::ReadData,
            &config,
            engine,
        )?;
        Ok(Self {
            snapshot: s.snapshot,
            engine: s.engine,
            name: s.name,
            config,
        })
    }

    fn execute_serial(&self) -> Result<(), Box<dyn std::error::Error>> {
        let scan = self.snapshot.clone().scan_builder().build()?;
        let result_iter = scan.execute(self.engine.clone())?;
        for result in result_iter {
            black_box(result?);
        }
        Ok(())
    }

    fn execute_parallel(&self, num_threads: usize) -> Result<(), Box<dyn std::error::Error>> {
        let scan = self.snapshot.clone().scan_builder().build()?;
        let table_root = scan.table_root().clone();
        let physical_schema = scan.physical_schema().clone();
        let logical_schema = scan.logical_schema().clone();

        let mut phase1 = scan.parallel_scan_metadata(self.engine.clone())?;
        for result in phase1.by_ref() {
            let scan_metadata = result?;
            read_scan_metadata_data(
                self.engine.as_ref(),
                scan_metadata,
                &table_root,
                &physical_schema,
                &logical_schema,
            )?;
        }

        match phase1.finish()? {
            AfterSequential::Done(_) => {}
            AfterSequential::Parallel { processor, files } => {
                if num_threads == 0 {
                    return Err("num_threads in ReadConfig must be greater than 0".into());
                }
                let files_per_worker = files.len().div_ceil(num_threads);

                let partitions: Vec<_> = files
                    .chunks(files_per_worker)
                    .map(|chunk| chunk.to_vec())
                    .collect();

                let processor = Arc::new(processor);

                let handles: Vec<_> = partitions
                    .into_iter()
                    .map(|partition_files| {
                        let engine = self.engine.clone();
                        let processor = processor.clone();
                        let table_root = table_root.clone();
                        let physical_schema = physical_schema.clone();
                        let logical_schema = logical_schema.clone();

                        thread::spawn(move || -> Result<(), crate::Error> {
                            if partition_files.is_empty() {
                                return Ok(());
                            }

                            let parallel =
                                ParallelPhase::try_new(engine.clone(), processor, partition_files)?;
                            for result in parallel {
                                let scan_metadata = result?;
                                read_scan_metadata_data(
                                    engine.as_ref(),
                                    scan_metadata,
                                    &table_root,
                                    &physical_schema,
                                    &logical_schema,
                                )?;
                            }

                            Ok(())
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().map_err(|_| -> Box<dyn std::error::Error> {
                        "Worker thread panicked".into()
                    })??;
                }
            }
        }
        Ok(())
    }
}

impl WorkloadRunner for ReadDataRunner {
    fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        match &self.config.parallel_scan {
            ParallelScan::Disabled => {
                self.execute_serial()?;
            }
            ParallelScan::Enabled { num_threads } => {
                self.execute_parallel(*num_threads)?;
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

fn read_scan_metadata_data(
    engine: &dyn Engine,
    scan_metadata: ScanMetadata,
    table_root: &Url,
    physical_schema: &SchemaRef,
    logical_schema: &SchemaRef,
) -> crate::DeltaResult<()> {
    fn scan_file_callback(batches: &mut Vec<state::ScanFile>, file: state::ScanFile) {
        batches.push(file);
    }

    let scan_files = scan_metadata.visit_scan_files(vec![], scan_file_callback)?;

    for scan_file in scan_files {
        let file_path = table_root.join(&scan_file.path)?;
        let mut selection_vector = scan_file.dv_info.get_selection_vector(engine, table_root)?;
        let meta = FileMeta {
            last_modified: 0,
            size: scan_file.size.try_into().map_err(|_| {
                crate::Error::generic("Unable to convert scan file size into FileSize")
            })?,
            location: file_path,
        };

        let read_result_iter =
            engine
                .parquet_handler()
                .read_parquet_files(&[meta], physical_schema.clone(), None)?;

        for read_result in read_result_iter {
            let read_result = read_result?;
            let logical = state::transform_to_logical(
                engine,
                read_result,
                physical_schema,
                logical_schema,
                scan_file.transform.clone(),
            );
            let len = logical.as_ref().map_or(0, |res| res.len());
            let mut sv = selection_vector.take();
            let rest = split_vector(sv.as_mut(), len, None);
            let result = match sv {
                Some(sv) => logical.and_then(|data| data.apply_selection_vector(sv)),
                None => logical,
            };
            selection_vector = rest;
            black_box(result?);
        }
    }

    Ok(())
}

pub fn create_read_runner(
    table_info: &TableInfo,
    case_name: &str,
    read_spec: &ReadSpec,
    operation: ReadOperation,
    config: ReadConfig,
    engine: Arc<dyn Engine>,
) -> Result<Box<dyn WorkloadRunner>, Box<dyn std::error::Error>> {
    match operation {
        ReadOperation::ReadMetadata => Ok(Box::new(ReadMetadataRunner::setup(
            table_info, case_name, read_spec, config, engine,
        )?)),
        ReadOperation::ReadData => Ok(Box::new(ReadDataRunner::setup(
            table_info, case_name, read_spec, config, engine,
        )?)),
    }
}

pub struct SnapshotConstructionRunner {
    url: Url,
    version: Option<u64>,
    engine: Arc<dyn Engine>,
    name: String,
}

impl SnapshotConstructionRunner {
    pub fn setup(
        table_info: &TableInfo,
        case_name: &str,
        snapshot_spec: &SnapshotConstructionSpec,
        engine: Arc<dyn Engine>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let table_root = table_info.resolved_table_root();
        let url = crate::try_parse_uri(table_root)?;

        let name = format!(
            "{}/{}/{}",
            table_info.name,
            case_name,
            snapshot_spec.as_str()
        );

        Ok(Self {
            url,
            version: snapshot_spec.version,
            engine,
            name,
        })
    }
}

impl WorkloadRunner for SnapshotConstructionRunner {
    fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut builder = Snapshot::builder_for(self.url.clone());
        if let Some(version) = self.version {
            builder = builder.at_version(version);
        }
        black_box(builder.build(self.engine.as_ref())?);

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::models::{ParallelScan, ReadConfig, ReadSpec, TableInfo};
    use crate::engine::sync::SyncEngine;
    use std::path::PathBuf;

    fn test_table_info() -> TableInfo {
        TableInfo {
            name: "basic_partitioned".to_string(),
            description: None,
            table_path: Some(format!(
                "{}/tests/data/basic_partitioned",
                env!("CARGO_MANIFEST_DIR")
            )),
            table_info_dir: PathBuf::new(),
        }
    }

    fn test_read_spec() -> ReadSpec {
        ReadSpec { version: None }
    }

    fn serial_config() -> ReadConfig {
        ReadConfig {
            name: "serial".to_string(),
            parallel_scan: ParallelScan::Disabled,
        }
    }

    fn parallel_config() -> ReadConfig {
        ReadConfig {
            name: "parallel".to_string(),
            parallel_scan: ParallelScan::Enabled { num_threads: 2 },
        }
    }

    fn test_engine() -> Arc<dyn Engine> {
        Arc::new(SyncEngine::new())
    }

    #[test]
    fn test_read_metadata_runner_serial() {
        let runner = ReadMetadataRunner::setup(
            &test_table_info(),
            "test_case",
            &test_read_spec(),
            serial_config(),
            test_engine(),
        )
        .expect("setup should succeed");
        assert_eq!(
            runner.name(),
            "basic_partitioned/test_case/read_metadata/serial"
        );
        assert!(runner.execute().is_ok());
    }

    #[test]
    fn test_read_metadata_runner_parallel() {
        let runner = ReadMetadataRunner::setup(
            &test_table_info(),
            "test_case",
            &test_read_spec(),
            parallel_config(),
            test_engine(),
        )
        .expect("setup should succeed");
        assert_eq!(
            runner.name(),
            "basic_partitioned/test_case/read_metadata/parallel"
        );
        assert!(runner.execute().is_ok());
    }

    fn test_snapshot_spec() -> SnapshotConstructionSpec {
        SnapshotConstructionSpec { version: None }
    }

    #[test]
    fn test_snapshot_construction_runner_setup() {
        let runner = SnapshotConstructionRunner::setup(
            &test_table_info(),
            "test_case",
            &test_snapshot_spec(),
            test_engine(),
        );
        assert!(runner.is_ok());
    }

    #[test]
    fn test_snapshot_construction_runner_name() {
        let runner = SnapshotConstructionRunner::setup(
            &test_table_info(),
            "test_case",
            &test_snapshot_spec(),
            test_engine(),
        )
        .expect("setup should succeed");
        assert_eq!(
            runner.name(),
            "basic_partitioned/test_case/snapshot_construction"
        );
    }

    #[test]
    fn test_snapshot_construction_runner_execute() {
        let runner = SnapshotConstructionRunner::setup(
            &test_table_info(),
            "test_case",
            &test_snapshot_spec(),
            test_engine(),
        )
        .expect("setup should succeed");
        assert!(runner.execute().is_ok());
    }

    #[test]
    fn test_create_read_runner_read_metadata() {
        let runner = create_read_runner(
            &test_table_info(),
            "test_case",
            &test_read_spec(),
            ReadOperation::ReadMetadata,
            serial_config(),
            test_engine(),
        )
        .expect("create_read_runner should succeed");
        assert!(runner.execute().is_ok());
    }

    #[test]
    fn test_create_read_runner_read_data() {
        let runner = create_read_runner(
            &test_table_info(),
            "test_case",
            &test_read_spec(),
            ReadOperation::ReadData,
            serial_config(),
            test_engine(),
        )
        .expect("create_read_runner should succeed");
        assert!(runner.execute().is_ok());
    }

    #[test]
    fn test_read_data_runner_serial() {
        let runner = ReadDataRunner::setup(
            &test_table_info(),
            "test_case",
            &test_read_spec(),
            serial_config(),
            test_engine(),
        )
        .expect("setup should succeed");
        assert_eq!(runner.name(), "basic_partitioned/test_case/read_data/serial");
        assert!(runner.execute().is_ok());
    }

    #[test]
    fn test_read_data_runner_parallel() {
        let runner = ReadDataRunner::setup(
            &test_table_info(),
            "test_case",
            &test_read_spec(),
            parallel_config(),
            test_engine(),
        )
        .expect("setup should succeed");
        assert_eq!(
            runner.name(),
            "basic_partitioned/test_case/read_data/parallel"
        );
        assert!(runner.execute().is_ok());
    }
}
