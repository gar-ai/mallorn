//! Mallorn CLI - Command-line interface for edge model delta updates

mod chain;
mod convert;
mod dict;
mod diff;
mod download;
mod fingerprint;
mod info;
mod patch;
mod sign;
mod verify;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mallorn")]
#[command(author, version, about = "Edge model delta updates for ML models")]
#[command(long_about = "Mallorn creates minimal delta patches for ML models, reducing OTA bandwidth by 95%+.\n\nSupported formats: TFLite, GGUF, ONNX, SafeTensors, OpenVINO, CoreML, TensorRT")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a delta patch between two models
    Diff {
        /// Source model file (the model you're updating FROM)
        old: PathBuf,

        /// Target model file (the model you're updating TO)
        new: PathBuf,

        /// Output patch file
        #[arg(short, long)]
        output: PathBuf,

        /// Compression level (1-22 for zstd, higher = smaller but slower)
        #[arg(long, default_value = "3")]
        level: i32,

        /// Enable neural-aware compression (experimental, better ratios)
        #[arg(long)]
        neural: bool,

        /// Pre-trained dictionary for improved compression
        #[arg(long, value_name = "FILE")]
        dict: Option<PathBuf>,

        /// Enable parallel tensor compression (faster on multi-core)
        #[arg(long)]
        parallel: bool,
    },

    /// Apply a patch to a model
    Patch {
        /// Source model file
        model: PathBuf,

        /// Patch file to apply
        patch: PathBuf,

        /// Output model file
        #[arg(short, long)]
        output: PathBuf,

        /// Use streaming mode (low memory, for large models)
        #[arg(long)]
        streaming: bool,

        /// Buffer size for streaming mode (default: 64MB)
        #[arg(long, default_value = "67108864")]
        buffer_size: usize,
    },

    /// Verify a patch can be applied to a model
    Verify {
        /// Source model file
        model: PathBuf,

        /// Patch file to verify
        patch: PathBuf,
    },

    /// Show information about a patch file
    Info {
        /// Patch file
        patch: PathBuf,

        /// Show detailed tensor-level information
        #[arg(long)]
        verbose: bool,
    },

    /// Convert a patch to streaming format for embedded devices
    Convert {
        /// Input patch file (.tflp or .ggup)
        input: PathBuf,

        /// Output streaming patch file (.mllp)
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Generate a new ED25519 keypair for signing
    Keygen {
        /// Output path for private key
        #[arg(long, default_value = "mallorn.key")]
        private_key: PathBuf,

        /// Output path for public key
        #[arg(long, default_value = "mallorn.pub")]
        public_key: PathBuf,
    },

    /// Sign a patch file with ED25519
    Sign {
        /// Patch file to sign
        patch: PathBuf,

        /// Private key file
        #[arg(short, long)]
        key: PathBuf,

        /// Output signed patch file
        #[arg(short, long)]
        output: PathBuf,

        /// Version number for downgrade protection
        #[arg(long, default_value = "1")]
        version: u64,
    },

    /// Verify a signed patch file
    VerifySignature {
        /// Signed patch file
        patch: PathBuf,

        /// Public key file (optional, uses embedded key if not provided)
        #[arg(short, long)]
        key: Option<PathBuf>,

        /// Minimum version to accept (for downgrade protection)
        #[arg(long)]
        min_version: Option<u64>,
    },

    /// Manage patch chains for incremental updates
    Chain {
        #[command(subcommand)]
        command: ChainCommands,
    },

    /// Manage compression dictionaries for improved compression ratios
    Dict {
        #[command(subcommand)]
        command: DictCommands,
    },

    /// Generate a quick fingerprint for a model file
    Fingerprint {
        /// Model file to fingerprint
        model: PathBuf,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Check against fingerprint database
        #[arg(long, value_name = "FILE")]
        db: Option<PathBuf>,

        /// Compare with another model
        #[arg(long, value_name = "FILE")]
        compare: Option<PathBuf>,

        /// Add to database with version string
        #[arg(long, value_name = "VERSION")]
        add_to_db: Option<String>,

        /// Database file for --add-to-db
        #[arg(long, value_name = "FILE", requires = "add_to_db")]
        db_file: Option<PathBuf>,
    },

    /// Download patches from a URL with resume support
    Download {
        /// URL to download from
        url: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Enable resume for interrupted downloads
        #[arg(long)]
        resume: bool,

        /// Expected SHA256 hash (hex) for verification
        #[arg(long, value_name = "HASH")]
        verify: Option<String>,

        /// Cache directory for partial downloads
        #[arg(long, value_name = "DIR")]
        cache: Option<PathBuf>,

        /// Fetch header only (for inspection)
        #[arg(long)]
        header_only: bool,

        /// Fetch manifest from URL
        #[arg(long)]
        manifest: bool,
    },
}

#[derive(Subcommand)]
enum DictCommands {
    /// Train a dictionary from model samples
    Train {
        /// Sample model files to train from
        #[arg(required = true)]
        samples: Vec<PathBuf>,

        /// Output dictionary file (.dict)
        #[arg(short, long)]
        output: PathBuf,

        /// Maximum dictionary size in bytes (default: 112KB)
        #[arg(long, default_value = "114688")]
        max_size: usize,
    },

    /// Show information about a dictionary file
    Info {
        /// Dictionary file
        dict: PathBuf,
    },
}

#[derive(Subcommand)]
enum ChainCommands {
    /// Create a new patch chain from an initial patch
    Create {
        /// Initial patch file to start the chain
        patch: PathBuf,

        /// Output chain file (.mchn)
        #[arg(short, long)]
        output: PathBuf,

        /// Optional chain identifier
        #[arg(long)]
        chain_id: Option<String>,
    },

    /// Append a patch to an existing chain
    Append {
        /// Existing chain file
        chain: PathBuf,

        /// Patch file to append
        patch: PathBuf,

        /// Output chain file (defaults to overwriting input)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Show information about a patch chain
    Info {
        /// Chain file
        chain: PathBuf,

        /// Show detailed link information
        #[arg(long)]
        verbose: bool,
    },

    /// Extract patches from a chain
    Extract {
        /// Chain file
        chain: PathBuf,

        /// Output directory for extracted patches
        #[arg(short, long)]
        output: PathBuf,

        /// Start hash (extract from this version)
        #[arg(long)]
        from: Option<String>,

        /// End hash (extract up to this version)
        #[arg(long)]
        to: Option<String>,
    },

    /// Squash multiple patches in a chain into a single patch
    Squash {
        /// Chain file to squash
        chain: PathBuf,

        /// Output patch file
        #[arg(short, long)]
        output: PathBuf,

        /// Start hash (squash from this version)
        #[arg(long)]
        from: Option<String>,

        /// End hash (squash up to this version)
        #[arg(long)]
        to: Option<String>,
    },

    /// Apply a chain to a model (update through multiple versions)
    Apply {
        /// Base model file
        model: PathBuf,

        /// Chain file
        chain: PathBuf,

        /// Output model file
        #[arg(short, long)]
        output: PathBuf,

        /// Target hash (stop at this version, defaults to head)
        #[arg(long)]
        target: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Diff {
            old,
            new,
            output,
            level,
            neural,
            dict,
            parallel,
        } => diff::run(&old, &new, &output, level, neural, dict.as_deref(), parallel),

        Commands::Patch {
            model,
            patch,
            output,
            streaming,
            buffer_size,
        } => patch::run(&model, &patch, &output, streaming, buffer_size),

        Commands::Verify { model, patch } => verify::run(&model, &patch),

        Commands::Info { patch, verbose } => info::run(&patch, verbose),

        Commands::Convert { input, output } => convert::run(&input, &output),

        Commands::Keygen {
            private_key,
            public_key,
        } => sign::generate(&private_key, &public_key),

        Commands::Sign {
            patch,
            key,
            output,
            version,
        } => sign::sign(&patch, &key, &output, version),

        Commands::VerifySignature {
            patch,
            key,
            min_version,
        } => sign::verify_signature(&patch, key.as_deref(), min_version),

        Commands::Chain { command } => match command {
            ChainCommands::Create {
                patch,
                output,
                chain_id,
            } => chain::create(&patch, &output, chain_id.as_deref()),

            ChainCommands::Append {
                chain: chain_path,
                patch,
                output,
            } => chain::append(&chain_path, &patch, output.as_deref()),

            ChainCommands::Info { chain, verbose } => chain::info(&chain, verbose),

            ChainCommands::Extract {
                chain: chain_path,
                output,
                from,
                to,
            } => chain::extract(&chain_path, &output, from.as_deref(), to.as_deref()),

            ChainCommands::Squash {
                chain: chain_path,
                output,
                from,
                to,
            } => chain::squash(&chain_path, &output, from.as_deref(), to.as_deref()),

            ChainCommands::Apply {
                model,
                chain: chain_path,
                output,
                target,
            } => chain::apply(&model, &chain_path, &output, target.as_deref()),
        },

        Commands::Dict { command } => match command {
            DictCommands::Train {
                samples,
                output,
                max_size,
            } => {
                let sample_refs: Vec<&std::path::Path> =
                    samples.iter().map(|p| p.as_path()).collect();
                dict::train(&sample_refs, &output, max_size)
            }

            DictCommands::Info { dict: dict_path } => dict::info(&dict_path),
        },

        Commands::Fingerprint {
            model,
            json,
            db,
            compare,
            add_to_db,
            db_file,
        } => {
            if let Some(other) = compare {
                fingerprint::compare(&model, &other)
            } else if let Some(version) = add_to_db {
                let db_path = db_file.unwrap_or_else(|| PathBuf::from("fingerprints.json"));
                fingerprint::add_to_db(&model, &db_path, &version)
            } else {
                fingerprint::run(&model, json, db.as_deref())
            }
        }

        Commands::Download {
            url,
            output,
            resume,
            verify,
            cache,
            header_only,
            manifest,
        } => {
            if header_only {
                download::header(&url)
            } else if manifest {
                download::manifest(&url, false)
            } else {
                download::run(&url, &output, resume, verify.as_deref(), cache.as_deref())
            }
        }
    }
}
