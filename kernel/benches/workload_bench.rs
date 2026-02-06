use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::Arc;
use std::hint::black_box;
use std::time::{Duration, Instant};
use std::fs;


use delta_kernel::benchmarks::models::{ParallelScan, ReadConfig, ReadOperation, Spec, TableInfo, WorkloadSpecVariant};
use delta_kernel::benchmarks::runners::{ReadMetadataRunner, WorkloadRunner};
use delta_kernel::engine::default::{DefaultEngine, DefaultEngineBuilder};
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::try_parse_uri;

use test_utils::load_test_data;
use tempfile::TempDir;

fn setup() -> (TempDir, String, Arc<DefaultEngine<TokioBackgroundExecutor>>) {
    // note this table _only_ has a _delta_log, no data files (can only do metadata reads)
    let table = "300k-add-files-100-col-partitioned";
    //let table2 = "table_2000_large_commits";
    let tempdir = load_test_data("./tests/data", table).unwrap();
    let table_path = tempdir.path().join(table);
    let table_path_str = table_path.to_str().unwrap().to_string();
    let url = try_parse_uri(&table_path_str).expect("Failed to parse table path");
    // TODO: use multi-threaded executor
    use delta_kernel::engine::default::storage::store_from_url;
    let store = store_from_url(&url).expect("Failed to create store");
    let engine = DefaultEngineBuilder::new(store).build();

    (tempdir, table_path_str, Arc::new(engine))
}

fn workload_benchmark(c: &mut Criterion) {
    let (_tempdir, table_path_str, engine) = setup();

    //get spec - will read from file so the op and config fields will be None

    let table_info_json = fs::read_to_string("tests/data/table_info.json")
        .expect("Failed to read table_info.json");
    let table_info = TableInfo::from_json(table_info_json).expect("Failed to parse table info");
    let spec_json = fs::read_to_string("tests/data/read_spec.json")
        .expect("Failed to read read_spec.json");
    let spec_json_v0 = fs::read_to_string("tests/data/read_spec_v0.json")
        .expect("Failed to read read_spec_v0.json");
    // let spec_json_v2 = fs::read_to_string("tests/data/read_spec_v2.json")
    //     .expect("Failed to read read_spec_v2.json");
    let spec = WorkloadSpecVariant::from_json(spec_json).expect("Failed to parse spec");
    let spec_v0 = WorkloadSpecVariant::from_json(spec_json_v0).expect("Failed to parse spec_v0");
    // let spec_v2 = WorkloadSpecVariant::from_json(spec_json_v2).expect("Failed to parse spec_v2");
    let workload_spec_variant = WorkloadSpecVariant {
        table_info: table_info.clone(),
        case_name: "test_case".to_string(),
        spec,
        operation: Some(ReadOperation::ReadMetadata),
        config: Some(ReadConfig {
            name: "serial".into(),
            parallel_scan: ParallelScan::Disabled,
        }),
        table_path: Some(table_path_str.clone()),
    };
    let workload_spec_variant_v0 = WorkloadSpecVariant {
        table_info: table_info.clone(),
        case_name: "test_case".to_string(),
        spec: spec_v0,
        operation: Some(ReadOperation::ReadMetadata),
        config: Some(ReadConfig {
            name: "serial".into(),
            parallel_scan: ParallelScan::Disabled,
        }),
        table_path: Some(table_path_str.clone()),
    };
    // let workload_spec_variant_v2 = WorkloadSpecVariant {
    //     table_info,
    //     case_name: "test_case".to_string(),
    //     spec: spec_v2,
    //     operation: Some(ReadOperation::ReadMetadata),
    //     config: Some(ReadConfig {
    //         name: "serial".into(),
    //         parallel_scan: ParallelScan::Disabled,
    //     }),
    //     table_path: Some(table_path_str),
    // };
    let mut group = c.benchmark_group("workload");

    let variants = [
        ("read", workload_spec_variant),
        // ("read_v2", workload_spec_variant_v2),
        ("read_v0", workload_spec_variant_v0),
    ];

    for (name, variant) in &variants {
        match &variant.spec {
            Spec::Read { version: _ } => {
                let variant = variant.clone();
                let engine = engine.clone();
                group.bench_function(*name, |b| {
                    b.iter_custom(|iters| {
                        let mut total_duration = Duration::ZERO;

                        for _ in 0..iters {
                            let mut runner = ReadMetadataRunner::new(
                                variant.clone(),
                                engine.clone()
                            );
                            runner.setup().expect("Setup failed during benchmark");

                            let start = Instant::now();
                            black_box(runner.execute().expect("Execution failed during benchmark"));
                            total_duration += start.elapsed();
                        }

                        total_duration
                    })
                });
            }
        }
    }

    group.finish();
}

criterion_group!(benches, workload_benchmark);
criterion_main!(benches);
