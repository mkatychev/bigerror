use derive_more as dm;

#[cfg(not(feature = "std"))]
use core::time::Duration;
#[cfg(feature = "std")]
use std::{path::Path, time::Duration};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format};

use error_stack::Context;

use crate::{
    Index, Report, ThinContext,
    attachment::{self, Display, FromTo, Unsupported, simple_type_name},
    ty,
};

#[cfg(feature = "std")]
use crate::AttachExt;

use crate::{Field, attachment::DisplayDuration};

/// Used to encapsulate opaque `dyn core::error::Error` types.
/// 
/// This wrapper allows you to work with arbitrary error types in a uniform way
/// while preserving the original error information.
#[derive(Debug, dm::Display)]
#[display("{_0}")]
pub struct BoxError(Box<dyn core::error::Error + 'static + Send + Sync>);
impl ::core::error::Error for BoxError {}

/// Represents errors emitted while processing bytes into an object.
/// 
/// This error type is commonly used by:
/// * Codecs, serializers, and deserializers
/// * Byte types such as `&[u8]`, `bytes::Bytes`, and `Vec<u8>`
/// 
/// Examples of types/traits that can emit decode errors:
/// * [tonic::codec::Encoder](https://docs.rs/tonic/latest/tonic/codec/trait.Encoder.html)
/// * [rkyv AllocSerializer](https://docs.rs/rkyv/latest/rkyv/ser/serializers/type.AllocSerializer.html)
/// * [serde::Serializer](https://docs.rs/serde/latest/serde/trait.Serializer.html)
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct DecodeError;

/// Represents errors emitted while turning an object into bytes.
/// 
/// This is the counterpart to [`DecodeError`] and is used in serialization,
/// encoding, and similar byte-generation operations.
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct EncodeError;

/// Represents errors emitted during authorization or verification checks.
/// 
/// This error type is used for authentication failures, permission denials,
/// token validation errors, and similar security-related issues.
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct AuthError;

/// Represents errors related to network operations.
/// 
/// This includes connection failures, timeouts, DNS resolution errors,
/// and other network-related issues.
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct NetworkError;

/// Represents errors emitted while processing strings (UTF-8 or otherwise).
/// 
/// This error type is commonly associated with:
/// * The [`std::str::FromStr`] trait
/// * The `.parse::<T>()` method
/// * String validation and formatting operations
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct ParseError;

/// Represents the conversion of an `Option<T>::None` into a [`Report`].
/// 
/// This error type is used when an expected value is missing, such as:
/// * Missing keys in maps or dictionaries
/// * Missing fields in data structures
/// * Failed lookups or searches
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct NotFound;

/// Represents errors related to database operations.
/// 
/// This includes connection errors, query failures, transaction issues,
/// and other database-related problems.
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct DbError;

/// Represents errors related to filesystem operations.
/// 
/// This includes file I/O errors, permission issues, path problems,
/// and other filesystem-related operations such as those in [`std::fs`].
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct FsError;

/// Represents errors emitted during the startup or provisioning phase of a program.
/// 
/// This includes configuration loading, resource initialization,
/// dependency setup, and other bootstrap-related failures.
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct SetupError;

/// Represents errors emitted during transformations between complex data types.
/// 
/// This error type is used for conversions between [non-scalar types](https://en.wikipedia.org/w/index.php?title=Scalar_processor&useskin=vector#Scalar_data_type)
/// such as structs, enums, and unions. It's commonly used in type conversion,
/// data mapping, and transformation operations.
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct ConversionError;

/// Represents errors for invalid input data or parameters.
/// 
/// This error type is used when input validation fails, including:
/// * Invalid parameter values
/// * Malformed input data
/// * Out-of-range values
/// * Format violations
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct InvalidInput;

/// Represents errors for invalid status conditions.
/// 
/// This error type is used when an operation encounters an unexpected
/// or invalid status, such as state machine violations or status checks.
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct InvalidStatus;

/// Represents errors for invalid state conditions.
/// 
/// This error type is used when an object or system is in an invalid state
/// for the requested operation, including state machine violations and
/// precondition failures.
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct InvalidState;

/// Represents errors related to configuration issues.
/// 
/// This error type is emitted during runtime and indicates problems with:
/// * Configuration file parsing
/// * Invalid configuration values
/// * Missing required settings
/// * Default setting conflicts
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct ConfigError;

/// Represents errors related to build processes.
/// 
/// This error type is typically emitted by:
/// * `build.rs` script failures
/// * Compilation errors
/// * Build tool issues
/// * Asset generation problems
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct BuildError;

/// Represents a timeout error with duration information.
/// 
/// This error type wraps a [`Duration`] and provides formatted output
/// showing the timeout value in a human-readable format.
#[derive(Debug, dm::Display)]
#[display("{}", Field::new("timeout", DisplayDuration(self.0)))]
pub struct Timeout(pub Duration);
impl ::core::error::Error for Timeout {}

/// Represents errors from failed assertions or invariant violations.
/// 
/// This error type is used for assertion failures, contract violations,
/// and other conditions that should never occur in correct program execution.
#[derive(ThinContext)]
#[bigerror(crate)]
pub struct AssertionError;

/// Trait for core error types that can be used in error reporting.
/// 
/// This trait combines the essential bounds needed for error types:
/// `Debug`, `Display`, `Send`, `Sync`, and `'static`.
pub trait CoreError: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static {}

impl<T> CoreError for T where T: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static {}

impl BoxError {
    /// Create a new `BoxError` report from any error type.
    /// 
    /// This method boxes the error and wraps it in a `Report`.
    #[track_caller]
    pub fn new<E>(err: E) -> Report<Self>
    where
        E: core::error::Error + 'static + Send + Sync,
    {
        Report::new(Self(Box::new(err)))
    }

    /// Create a `BoxError` report from an already-boxed error.
    #[track_caller]
    pub fn from(err: Box<dyn core::error::Error + 'static + Send + Sync>) -> Report<Self> {
        Report::new(Self(err))
    }
}

impl FsError {
    /// Create a filesystem error with path information.
    /// 
    /// This method creates an `FsError` report and attaches the path
    /// that caused the error for better debugging.
    #[cfg(feature = "std")]
    #[track_caller]
    pub fn with_path(path: impl AsRef<Path>) -> Report<Self> {
        let path = path.as_ref().display().to_string();
        Report::new(Self).attach_kv("path", path)
    }
}

impl InvalidInput {
    /// Create an invalid input error with path information.
    /// 
    /// This method creates an `InvalidInput` report and attaches the path
    /// that contained the invalid input.
    #[cfg(feature = "std")]
    #[track_caller]
    pub fn with_path(path: impl AsRef<Path>) -> Report<Self> {
        let path = path.as_ref().display().to_string();
        Report::new(Self).attach_kv("path", path)
    }

    /// Create an invalid input error with type information.
    /// 
    /// This method creates an `InvalidInput` report and attaches the type
    /// name that was invalid.
    #[track_caller]
    pub fn type_name<T: ?Sized>() -> Report<Self> {
        let type_name = simple_type_name::<T>();
        Report::new(Self).attach_printable(format!("type: {type_name}"))
    }

    /// Create an invalid input error for unsupported operations.
    /// 
    /// This method creates an `InvalidInput` report indicating that
    /// the requested operation is not supported.
    #[track_caller]
    pub fn unsupported() -> Report<Self> {
        Report::new(Self).attach_printable(Unsupported)
    }
}

impl ConversionError {
    /// Create a new conversion error with type information.
    /// 
    /// This method creates a `ConversionError` report showing the source
    /// and target types of the failed conversion.
    #[track_caller]
    pub fn new<F, T>() -> Report<Self> {
        Report::new(Self).attach_printable(FromTo(ty!(F), ty!(T)))
    }
    
    /// Create a conversion error from an existing context with type information.
    /// 
    /// This method converts an existing error context into a `ConversionError`
    /// and attaches the source and target type information.
    #[track_caller]
    pub fn from<F, T>(ctx: impl Context) -> Report<Self> {
        Self::report(ctx).attach_printable(FromTo(ty!(F), ty!(T)))
    }
}

impl NotFound {
    /// Create a not found error for a missing field.
    /// 
    /// This method creates a `NotFound` report indicating that a specific
    /// field is missing from a data structure.
    #[track_caller]
    pub fn with_field(field: &'static str) -> Report<Self> {
        Report::new(Self).attach_printable(Field::new(field, attachment::Missing))
    }

    /// Create a not found error for a missing indexed item.
    /// 
    /// This method creates a `NotFound` report indicating that an item
    /// at a specific index or key could not be found.
    pub fn with_index<T, K: Display>(key: K) -> Report<Self> {
        Self::attach_kv(Index(key), ty!(T))
    }
}

impl ParseError {
    /// Create a parse error for an invalid field.
    /// 
    /// This method creates a `ParseError` report indicating that a specific
    /// field could not be parsed correctly.
    #[track_caller]
    pub fn with_field(field: &'static str) -> Report<Self> {
        Report::new(Self).attach_printable(Field::new(field, attachment::Invalid))
    }
}
