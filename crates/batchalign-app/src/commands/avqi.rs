//! Command-owned metadata for `avqi`.

use crate::ReleasedCommand;
use crate::commands::spec::declare_media_analysis_command;
use crate::worker::InferTask;

declare_media_analysis_command!(
    AvqiCommand,
    AVQI_DEFINITION,
    ReleasedCommand::Avqi,
    InferTask::Avqi,
    true,
);
