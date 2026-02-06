use crate::benchmarks::models::{Spec, WorkloadSpecVariant};
use crate::snapshot::Snapshot;
use crate::Engine;

use std::sync::Arc;

pub trait WorkloadRunner {
    fn setup(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    fn execute(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    fn name(&self) -> String;
}

pub struct ReadMetadataRunner {
    spec_variant: WorkloadSpecVariant,
    //config: ReadConfig,
    engine: Arc<dyn Engine>,
    snapshot: Option<Arc<Snapshot>>,
}

impl ReadMetadataRunner {
    pub fn new(spec_variant: WorkloadSpecVariant, engine: Arc<dyn Engine>) -> Self {
        Self {
            spec_variant,
            //config,
            engine,
            snapshot: None,
        }
    }
}

impl WorkloadRunner for ReadMetadataRunner {
    fn setup(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Get table path from spec variant, or use default
        let table_path = self.spec_variant.table_path.as_ref()
            .ok_or("table_path not set in WorkloadSpecVariant")?;

        let url = crate::try_parse_uri(table_path)?;

        let version = match &self.spec_variant.spec {
            Spec::Read { version } => version,
        };

        let mut builder = Snapshot::builder_for(url);
        if let Some(version) = version {
            builder = builder.at_version(*version);
        }
        let snapshot = builder.build(self.engine.as_ref())?;

        self.snapshot = Some(snapshot);
        Ok(())
    }

    fn execute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let snapshot = self.snapshot.as_ref()
            .ok_or("Snapshot not initialized. Call setup() first.")?;

        // Build a scan over the snapshot
        let scan = snapshot.clone().scan_builder().build()?;

        // TODO: When parallel_scan_metadata() API is available (PR #1547), use it for distributed execution:
        // match &self.config.parallel_scan {
        //     ParallelScan::Disabled => {
        //         // Serial execution
        //         let scan_metadata_iter = scan.scan_metadata(self.engine.as_ref())?;
        //         for result in scan_metadata_iter {
        //             let _metadata = result?;
        //         }
        //     }
        //     ParallelScan::Enabled { num_threads } => {
        //         // Distributed execution (when API is available)
        //         let (phase1, files) = scan.parallel_scan_metadata(self.engine.as_ref())?;
        //         // Process phase1 results and distribute files across num_threads
        //         // ...
        //     }
        // }

        // For now, always use serial execution (only option available on this branch)
        let scan_metadata_iter = scan.scan_metadata(self.engine.as_ref())?;
        for result in scan_metadata_iter {
            let _metadata = result?;
            // Just consume the metadata - the work happens in the iteration
        }

        Ok(())
    }
    
    fn name(&self) -> String {
        self.spec_variant.full_name()
    }
}
