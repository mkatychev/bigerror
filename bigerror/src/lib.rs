//! Enhanced error handling library built on top of [`error-stack`][error_stack].
//!
//! `bigerror` provides ergonomic error handling adding out of the box functionality to [`error-stack`][error_stack]
//! for common scenarios.
//!
//! # Key Features
//!
//! - **Useful context attachments** - Key-value pairs, field status, type information
//!   - see [`kv!`], [`ty!`], [`expect_field!`]
//! - **Pre-defined contexts**: [`NotFound`], [`ParseError`], [`Timeout`], etc.
//! - **`no_std` support**: Works in embedded and constrained environments
//!
//! # Quick Start
//!
//! ```rust
//! use bigerror::{ThinContext, Report, expect_field, IntoContext, ResultIntoContext, NotFound};
//!
//! // Define your error type
//! #[derive(ThinContext)]
//! struct MyError;
//!
//! fn parse_number(input: &str) -> Result<i32, Report<MyError>> {
//!     // Use context conversion for error handling
//!     let num: i32 = input.parse()
//!         .into_ctx::<MyError>()?; // `::<MyError>` can be omitted
//!
//!     Ok(num)
//! }
//!
//! // Example with `expect_field` for optional values
//! fn get_config_value() -> Result<&'static str, Report<NotFound>> {
//!     let config = Some("production");
//!     expect_field!(config).into_ctx()
//! }
//! ```
//!
//! ## Attachments
//!
//! Attach contextual information to errors using various attachment types:
//!
//! ```rust
//! use bigerror::{ThinContext, kv, ty, KeyValue};
//!
//! #[derive(ThinContext)]
//! struct MyError;
//!
//! // Key-value attachments
//! let error = MyError::attach_kv("user_id", 42);
//!
//! // Type-value attachments
//! let data = vec![1, 2, 3];
//! let error = MyError::attach_kv(ty!(Vec<i32>), data.len());
//!
//! #[derive(Debug, Clone, Eq, PartialEq)]
//! struct Username(String);
//! impl std::fmt::Display for Username {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         write!(f, "{}", self.0)
//!     }
//! }
//!
//! let username = String::from("alice");
//! assert_eq!(kv!(username.clone()), KeyValue("username", String::from("alice")));
//! let error = MyError::attach(kv!(username.clone())); // "username": "alice"
//!
//! let username = Username(username);
//! assert_eq!(format!("{}", kv!(ty: username)), "<Username>: alice");
//! ```
//!
//! # Feature Flags
//!
//! - `std` (default) - Standard library support
//! - `backtrace` (default) - Backtrace support
//! - `tracing` - Integration with the tracing ecosystem
//! - `serde` - Serialization support
//! - `anyhow` - Compatibility with anyhow
//! - `eyre` - Compatibility with eyre
//!
//! # Pre-defined Error Contexts
//!
//! Common error contexts are provided out of the box:
//!
//! - [`NotFound`] - Missing resources, failed lookups
//! - [`ParseError`] - Parsing and deserialization failures
//! - [`Timeout`] - Operations that exceed time limits
//! - [`InvalidInput`] - Validation and input errors
//! - [`ConversionError`] - Type conversion failures
//!
//! See the [`context`] module for the complete list.

#![cfg_attr(not(feature = "std"), no_std)]
#![expect(deprecated, reason = "We use `Context` to maintain compatibility")]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec};

use error_stack::fmt::ColorMode;

#[cfg(all(not(feature = "std"), feature = "tracing"))]
use core::fmt;
#[cfg(not(feature = "std"))]
use core::{
    any::{self, TypeId},
    mem,
    panic::Location,
};
#[cfg(all(feature = "std", feature = "tracing"))]
use std::fmt;
#[cfg(feature = "std")]
use std::{
    any::{self, TypeId},
    mem,
    panic::Location,
    path::Path,
};
#[cfg(feature = "tracing")]
use tracing::{Level, debug, error, info, trace, warn};

/// Derive macro for implementing [`ThinContext`] trait on zero-sized error types.
pub use bigerror_derive::ThinContext;
/// Re-export of error-stack types and macros for convenience.
pub use error_stack::{self, Context, IntoReport, Report, ResultExt, bail, ensure, report};

/// Error attachment types and utilities for adding context to error reports.
pub mod attachment;
/// Common error context types for different categories of errors.
pub mod context;

/// Re-export of commonly used attachment types.
pub use attachment::{Expectation, Field, Index, KeyValue, Type};

use attachment::{Dbg, Debug, Display};
/// Re-export of all error context types.
pub use context::*;

/// Initialize error reporting with colored output.
///
/// Sets the error-stack color mode to full color for enhanced readability
/// in terminals that support ANSI color codes.
pub fn init_colour() {
    Report::set_color_mode(ColorMode::Color);
}

/// Initialize error reporting with emphasis only (no full color).
///
/// Sets the error-stack color mode to use emphasis styling without
/// full color support, suitable for terminals with limited color support.
pub fn init_emphasis() {
    Report::set_color_mode(ColorMode::Emphasis);
}

/// Initialize error reporting with no ANSI formatting.
///
/// Disables all color and emphasis styling for plain text output,
/// suitable for logging or environments that don't support ANSI codes.
pub fn init_no_ansi() {
    Report::set_color_mode(ColorMode::None);
}

/// A trait for zero-sized error types that provides convenient error creation methods.
///
/// `ThinContext` extends the functionality of `error_stack::Context` by providing
/// static methods for creating error reports with attachments. This trait is ideally
/// used for zero-sized error types or types that hold only `'static` references.
///
/// # Example
///
/// ```
/// use bigerror::{ThinContext, Report};
///
/// #[derive(ThinContext)]
/// pub struct MyError;
///
/// // Create an error with an attachment
/// let error: Report<MyError> = MyError::attach("Something went wrong");
/// ```
pub trait ThinContext
where
    Self: Sized + Context,
{
    /// The singleton value for this zero-sized error type.
    const VALUE: Self;

    /// Create a new error report by converting from another context type.
    ///
    /// # Arguments
    /// * `ctx` - The source context to convert from
    fn report<C: Context>(ctx: C) -> Report<Self> {
        Report::new(ctx).change_context(Self::VALUE)
    }

    /// Create an error report with an attachment computed by a closure.
    ///
    /// This is useful for lazy evaluation of expensive attachment computations.
    #[track_caller]
    fn attach_with<A>(attachment: impl FnOnce() -> A) -> Report<Self>
    where
        A: Display,
    {
        Report::new(Self::VALUE).attach(attachment())
    }

    /// Create an error report with a displayable attachment.
    #[track_caller]
    fn attach<A>(value: A) -> Report<Self>
    where
        A: Display,
    {
        Report::new(Self::VALUE).attach(value)
    }
    /// Create an error report with a debug-formatted attachment.
    ///
    /// The attachment will be formatted using `Debug` instead of `Display`.
    #[track_caller]
    fn attach_dbg<A>(value: A) -> Report<Self>
    where
        A: Debug,
    {
        Self::attach(Dbg(value))
    }
    /// Create an error report with a key-value pair attachment.
    #[track_caller]
    fn attach_kv<K, V>(key: K, value: V) -> Report<Self>
    where
        K: Display,
        V: Display,
    {
        Self::attach(KeyValue(key, value))
    }
    /// Create an error report with a key-value pair where the value is debug-formatted.
    #[track_caller]
    fn attach_kv_dbg<K, V>(key: K, value: V) -> Report<Self>
    where
        K: Display,
        V: Debug,
    {
        Self::attach(KeyValue::dbg(key, value))
    }
    /// Create an error report with a field attachment.
    ///
    /// This represents a property or field of a data structure and its status.
    #[track_caller]
    fn attach_field<S: Display>(key: &'static str, status: S) -> Report<Self> {
        Self::attach(Field::new(key, status))
    }

    /// Create an error report showing expected vs actual values.
    ///
    /// Useful for validation errors where you want to show what was expected
    /// versus what was actually received.
    #[track_caller]
    fn expected_actual<A: attachment::Display>(expected: A, actual: A) -> Report<Self> {
        Self::attach(Expectation { expected, actual })
    }

    /// Create an error report with a type-value pair attachment.
    ///
    /// The key will be the type name and the value will be the provided value.
    #[track_caller]
    fn attach_ty_val<A: Display>(value: A) -> Report<Self> {
        Self::attach_kv(ty!(A), value)
    }

    /// Create an error report with a type-value pair where the value is debug-formatted.
    #[track_caller]
    fn attach_ty_dbg<A: Debug>(value: A) -> Report<Self> {
        Self::attach_kv_dbg(ty!(A), value)
    }

    /// Create an error report with just a type attachment.
    #[track_caller]
    fn attach_ty<A>() -> Report<Self> {
        Self::attach(ty!(A))
    }

    /// Create an error report with a type as field name and status.
    #[track_caller]
    fn attach_ty_status<A: Send + Sync + 'static>(status: impl Display) -> Report<Self> {
        Self::attach(Field::new(ty!(A), status))
    }
}

/// Trait for converting `Result<T, E>` into `Result<T, Report<C>>` with automatic error wrapping.
///
/// This trait extends the functionality of `error_stack::IntoReport` by allowing
/// implicit conversion from any error type to a specific context type with enhanced
/// error reporting that includes source chain information.
pub trait ReportAs<T> {
    /// Convert this result into a report with the specified context type.
    ///
    /// This method automatically converts any `Result<T, E>` where `E` implements
    /// `Context + Error` into a `Result<T, Report<C>>` where `C` is your desired
    /// context type.
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{ReportAs, ThinContext, Report};
    ///
    /// #[derive(ThinContext)]
    /// struct MyError;
    ///
    /// fn parse_number(s: &str) -> Result<i32, Report<MyError>> {
    ///     s.parse().report_as() // Converts ParseIntError automatically
    /// }
    ///
    /// // Error includes original ParseIntError details + type information
    /// let result = parse_number("not_a_number");
    /// assert!(result.is_err());
    /// ```
    fn report_as<C: ThinContext>(self) -> Result<T, Report<C>>;
}

impl<T, E: Context + core::error::Error> ReportAs<T> for Result<T, E> {
    #[inline]
    #[track_caller]
    fn report_as<C: ThinContext>(self) -> Result<T, Report<C>> {
        // TODO #[track_caller] on closure
        // https://github.com/rust-lang/rust/issues/87417
        // self.map_err(|e| Report::new(C::VALUE).attach(e))
        match self {
            Ok(v) => Ok(v),
            Err(e) => {
                let ty = any::type_name_of_val(&e);
                let mut external_report = Report::new(e).attach(ty);
                let mut curr_source = external_report.current_context().source();
                let mut child_errs = vec![];
                while let Some(child_err) = curr_source {
                    let new_err = format!("{child_err}");
                    // workaround comparison when the there is an
                    // error with a transparent inner type to avoid attaching
                    // the same message twice
                    if Some(&new_err) != child_errs.last() {
                        child_errs.push(new_err);
                    }
                    curr_source = child_err.source();
                }
                while let Some(child_err) = child_errs.pop() {
                    external_report = external_report.attach(child_err);
                }
                Err(external_report.into_ctx())
            }
        }
    }
}

/// Trait for converting error reports from one context type to another.
pub trait IntoContext {
    /// Convert this error report to use a different context type.
    ///
    /// This method transforms a `Report<C>` into a `Report<C2>` with `C2` being a [`ThinContext`].
    /// When the [`std::any::TypeId`] of `C` and `C2` is the same, only a [`Location`] is attached.
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{IntoContext, ThinContext, Report, NotFound, ResultIntoContext, DbError};
    ///
    /// #[derive(ThinContext)]
    /// struct ServiceError;
    ///
    /// fn fetch_user_data() -> Result<String, Report<ServiceError>> {
    ///     // 1. Simulate database operation that returns `NotFound`
    ///     let db_result: Result<String, Report<NotFound>> =
    ///         Err(NotFound::attach_kv("user_id", 123));
    ///
    ///     // 2. Convert `NotFound` to `DbError`
    ///     let database_result: Result<String, Report<DbError>> =
    ///         db_result.into_ctx();
    ///
    ///     // 3. Convert `DbError` into `ServiceError`
    ///     database_result.into_ctx()
    /// }
    /// ```
    fn into_ctx<C2: ThinContext>(self) -> Report<C2>;
}

impl<C: 'static> IntoContext for Report<C> {
    #[inline]
    #[track_caller]
    fn into_ctx<C2: ThinContext>(self) -> Report<C2> {
        if TypeId::of::<C>() == TypeId::of::<C2>() {
            // if C and C2 are zero-sized and have the same TypeId then they are covariant
            unsafe {
                return mem::transmute::<Self, Report<C2>>(self.attach_opaque(*Location::caller()));
            }
        }
        self.change_context(C2::VALUE)
    }
}

/// Extension trait for `Result<T, Report<C>>` that provides context conversion methods.
pub trait ResultIntoContext: ResultExt {
    /// Convert the error context type of this result.
    ///
    /// This method transforms a `Result<T, E>` into a `Result<T, Report<C>>`, converting
    /// any error type into a report with `C` being a [`ThinContext`]
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{ResultIntoContext, ThinContext, Report};
    ///
    /// #[derive(ThinContext)]
    /// struct MyParseError;
    ///
    /// fn parse_data() -> Result<i32, Report<MyParseError>> {
    ///     "42".parse().into_ctx() // Converts ParseIntError to Report<MyParseError>
    /// }
    ///
    /// assert_eq!(parse_data().unwrap(), 42);
    /// ```
    fn into_ctx<C2: ThinContext>(self) -> Result<Self::Ok, Report<C2>>;

    /// Chain operations while converting error context (similar to `Result::and_then`).
    ///
    /// This method allows you to chain operations that might fail, automatically converting
    /// any errors from the current result to the target context type before passing the
    /// success value to the closure. This is equivalent to `Result::and_then` but with
    /// automatic error context conversion.
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{
    ///     DbError, NotFound, OptionReport, Report, ResultIntoContext, expect_field,
    /// };
    ///
    /// fn find_user(id: u64) -> Result<User, Report<NotFound>> {
    ///     Some(User {
    ///         id,
    ///         email: Some("user@example.com".to_string()),
    ///     })
    ///     .expect_or()
    /// }
    /// struct User {
    ///     id: u64,
    ///     email: Option<String>,
    /// }
    ///
    /// fn find_user_and_get_email(user_id: u64) -> Result<String, Report<DbError>> {
    ///     // Simulate finding a user (might return NotFound)
    ///     find_user(user_id).and_then_ctx(|user| {
    ///         // Extract email from user
    ///         expect_field!(user.email).into_ctx()
    ///     })
    /// }
    /// ```
    fn and_then_ctx<U, F, C2>(self, op: F) -> Result<U, Report<C2>>
    where
        C2: ThinContext,
        F: FnOnce(Self::Ok) -> Result<U, Report<C2>>;

    /// Map the success value while converting error context (similar to `Result::map`).
    ///
    /// This method transforms the success value of a result using the provided closure,
    /// while automatically converting any errors to the target context type. This is
    /// equivalent to `Result::map` but with automatic error context conversion.
    ///
    /// # Example
    ///
    /// ```
    /// use std::path::Path;
    /// use bigerror::{ResultIntoContext, Report, FsError};
    ///
    /// fn count_lines_in_file(path: &Path) -> Result<usize, Report<FsError>> {
    ///     std::fs::read_to_string(path)
    ///         .map_ctx(|content| content.lines().count()) // Count lines and do conversion
    /// }
    /// ```
    fn map_ctx<U, F, C2>(self, op: F) -> Result<U, Report<C2>>
    where
        C2: ThinContext,
        F: FnOnce(Self::Ok) -> U;
}

impl<T, E: IntoReport> ResultIntoContext for Result<T, E>
where
    <E as IntoReport>::Context: Sized + 'static,
{
    #[inline]
    #[track_caller]
    fn into_ctx<C2: ThinContext>(self) -> Result<T, Report<C2>> {
        // Can't use `map_err` as `#[track_caller]` is unstable on closures
        match self {
            Ok(ok) => Ok(ok),
            Err(e) => Err(IntoContext::into_ctx(e.into_report())),
        }
    }

    #[inline]
    #[track_caller]
    fn and_then_ctx<U, F, C2>(self, op: F) -> Result<U, Report<C2>>
    where
        C2: ThinContext,
        F: FnOnce(T) -> Result<U, Report<C2>>,
    {
        match self {
            Ok(t) => op(t),
            Err(e) => Err(IntoContext::into_ctx(e.into_report())),
        }
    }

    #[inline]
    #[track_caller]
    fn map_ctx<U, F, C2>(self, op: F) -> Result<U, Report<C2>>
    where
        C2: ThinContext,
        F: FnOnce(T) -> U,
    {
        match self {
            Ok(t) => Ok(op(t)),
            Err(e) => Err(IntoContext::into_ctx(e.into_report())),
        }
    }
}

/// Extension trait that adds attachment methods to error reports and results.
///
/// This trait provides convenient methods for attaching various types of context
/// information to error reports.
pub trait AttachExt {
    /// Attach a key-value pair to the error.
    ///
    /// This method adds contextual information in the form of a key-value pair to the error
    /// report using the [`KeyValue`] type. Both the key and value must implement `Display` and will be formatted using
    /// their `Display` implementations in error messages.
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{AttachExt, ThinContext, Report};
    ///
    /// #[derive(ThinContext)]
    /// struct MyError;
    ///
    /// fn process_user(user_id: u64) -> Result<String, Report<MyError>> {
    ///     // Simulate an error with user context
    ///     Err(MyError::attach("processing failed"))
    ///         .attach_kv("user_id", user_id)
    ///         .attach_kv("operation", "data_processing")
    /// }
    /// ```
    #[must_use]
    fn attach_kv<K, V>(self, key: K, value: V) -> Self
    where
        K: Display,
        V: Display;

    /// Attach a key-value pair where the value is debug-formatted.
    ///
    /// This method is similar to [`attach_kv`](Self::attach_kv) but uses the `Debug`
    /// implementation of the value instead of `Display`. This is useful for types that
    /// don't implement `Display` or when you want the debug representation for
    /// diagnostic purposes.
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{AttachExt, ThinContext, Report};
    ///
    /// #[derive(ThinContext)]
    /// struct ValidationError;
    ///
    /// #[derive(Debug)]
    /// struct Config {
    ///     debug_mode: bool,
    ///     max_connections: usize,
    /// }
    ///
    /// fn validate_config(config: Config) -> Result<(), Report<ValidationError>> {
    ///     if config.max_connections == 0 {
    ///         return Err(ValidationError::attach("invalid max_connections"))
    ///             .attach_kv_dbg("config", config); // Uses Debug formatting
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    fn attach_kv_dbg<K, V>(self, key: K, value: V) -> Self
    where
        K: Display,
        V: Debug;

    /// Attach a field with its status.
    ///
    /// This method attaches information about a specific field or property and its status.
    /// It's particularly useful for validation errors, data processing failures, or when
    /// indicating the state of specific fields in data structures.
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{AttachExt, ThinContext, Report, attachment::Missing};
    ///
    /// #[derive(ThinContext)]
    /// struct ValidationError;
    ///
    /// fn validate_user_data(email: Option<&str>, age: Option<u32>) -> Result<(), Report<ValidationError>> {
    ///     let mut error = None;
    ///
    ///     if email.is_none() {
    ///         error = Some(ValidationError::attach("validation failed")
    ///             .attach_field_status("email", Missing));
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    fn attach_field_status<S>(self, name: &'static str, status: S) -> Self
    where
        S: Display;

    /// Attach a debug-formatted value.
    ///
    /// This method attaches a value to the error using its `Debug` implementation for
    /// formatting. This is useful for types that don't implement `Display` or when
    /// you want the detailed debug representation rather than the user-friendly display.
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{AttachExt, ThinContext, Report, DecodeError};
    ///
    /// fn decode_data(data: &mut Vec<u8>) -> Result<&str, Report<DecodeError>> {
    ///     if data.is_empty() {
    ///         return Err(DecodeError::attach("no data found"))
    ///             .attach_dbg(data.clone());
    ///     }
    ///
    ///     // ...
    ///
    ///     Ok("data processed")
    /// }
    /// ```
    #[must_use]
    fn attach_dbg<A>(self, value: A) -> Self
    where
        A: Debug;

    /// Attach a type-value pair where the type name is the key.
    ///
    /// This is a convenience method that creates a key-value attachment where the key is
    /// the type name (using the [`ty!`] macro) and the value is the provided value.
    /// This is useful for adding context about what type of value was involved in an error.
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{AttachExt, ThinContext, Report};
    ///
    /// #[derive(ThinContext)]
    /// struct ProcessingError;
    ///
    /// fn process_count(count: usize) -> Result<String, Report<ProcessingError>> {
    ///     if count == 0 {
    ///         return Err(ProcessingError::attach("invalid count"))
    ///             .attach_ty_val(count); // Attaches "<usize>: 0"
    ///     }
    ///     Ok(format!("Processing {} items", count))
    /// }
    /// ```
    #[must_use]
    fn attach_ty_val<A>(self, value: A) -> Self
    where
        Self: Sized,
        A: Display,
    {
        self.attach_kv(ty!(A), value)
    }

    /// Attach a file system path to the error.
    ///
    /// This method attaches a file system path as a key-value pair where the key is "path"
    /// and the value is the string representation of the path. This is commonly used for
    /// file I/O operations to provide context about which file caused the error.
    ///
    /// **Note**: The path is converted to a string immediately when this method is called
    /// (not lazily), so it's recommended to use this in `.map_err()` chains to avoid
    /// unnecessary string conversions when no error occurs.
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{AttachExt, ThinContext, Report, ResultIntoContext, FsError};
    /// use std::path::Path;
    ///
    /// fn read_config_file<P: AsRef<Path>>(path: P) -> Result<String, Report<FsError>> {
    ///     std::fs::read_to_string(&path)
    ///         .into_ctx()
    ///         .attach_path(path) // Attaches "path: /etc/config.toml"
    /// }
    /// ```
    #[must_use]
    #[cfg(feature = "std")]
    fn attach_path<P: AsRef<Path>>(self, path: P) -> Self
    where
        Self: Sized,
    {
        let path = path.as_ref().display().to_string();
        self.attach_kv("path", path)
    }
}

impl<C> AttachExt for Report<C> {
    #[inline]
    #[track_caller]
    fn attach_kv<K, V>(self, key: K, value: V) -> Self
    where
        K: Display,
        V: Display,
    {
        self.attach(KeyValue(key, value))
    }

    #[inline]
    #[track_caller]
    fn attach_kv_dbg<K, V>(self, key: K, value: V) -> Self
    where
        K: Display,
        V: Debug,
    {
        self.attach(KeyValue::dbg(key, value))
    }

    #[inline]
    #[track_caller]
    fn attach_field_status<S>(self, name: &'static str, status: S) -> Self
    where
        S: Display,
    {
        self.attach(Field::new(name, status))
    }

    #[inline]
    #[track_caller]
    fn attach_dbg<A>(self, value: A) -> Self
    where
        A: Debug,
    {
        self.attach(Dbg(value))
    }
}

impl<T, C> AttachExt for Result<T, Report<C>> {
    #[inline]
    #[track_caller]
    fn attach_kv<K, V>(self, key: K, value: V) -> Self
    where
        K: Display,
        V: Display,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(report) => Err(report.attach(KeyValue(key, value))),
        }
    }

    #[inline]
    #[track_caller]
    fn attach_kv_dbg<K, V>(self, key: K, value: V) -> Self
    where
        K: Display,
        V: Debug,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(report) => Err(report.attach(KeyValue::dbg(key, value))),
        }
    }

    #[inline]
    #[track_caller]
    fn attach_field_status<S>(self, name: &'static str, status: S) -> Self
    where
        S: Display,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(report) => Err(report.attach(Field::new(name, status))),
        }
    }

    #[inline]
    #[track_caller]
    fn attach_dbg<A>(self, value: A) -> Self
    where
        A: Debug,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(report) => Err(report.attach(Dbg(value))),
        }
    }
}

#[cfg(feature = "tracing")]
/// Trait for logging errors while maintaining functional programming patterns.
///
/// This trait provides methods for logging errors either by consuming them
/// or by logging and then forwarding them in a functional chain.
pub trait LogError<T, E>
where
    E: fmt::Debug,
{
    /// Log the error and consume the result (swallows the error).
    fn log_err(self);
    /// Log the error with an additional attachment and consume the result.
    fn log_attached_err<A>(self, attachment: A)
    where
        A: fmt::Debug + Send + Sync + 'static;
    /// Log the error at ERROR level and forward the result.
    fn and_log_err(self) -> Result<T, E>;
    /// Log the error at the specified level and forward the result.
    fn and_log(self, level: Level) -> Result<T, E>;
    /// Log the error with an attachment and forward the result.
    fn and_attached_err<A>(self, attachment: A) -> Result<T, E>
    where
        A: fmt::Debug + Send + Sync + 'static;
}

// TODO add log crate support
#[cfg(feature = "tracing")]
impl<T, E> LogError<T, E> for Result<T, E>
where
    E: fmt::Debug,
{
    #[inline]
    #[track_caller]
    fn log_err(self) {
        if let Err(e) = self {
            error!(message = ?e);
        }
    }

    #[inline]
    #[track_caller]
    fn log_attached_err<A>(self, attachment: A)
    where
        A: fmt::Debug + Send + Sync + 'static,
    {
        if let Err(e) = self {
            error!(err = ?e, "{attachment:?}");
        }
    }
    #[inline]
    #[track_caller]
    fn and_log(self, level: Level) -> Self {
        if let Err(err) = &self {
            match level {
                Level::TRACE => trace!(?err),
                Level::DEBUG => debug!(?err),
                Level::INFO => info!(?err),
                Level::WARN => warn!(?err),
                Level::ERROR => error!(?err),
            }
        }
        self
    }

    #[inline]
    #[track_caller]
    fn and_log_err(self) -> Self {
        if let Err(e) = &self {
            error!(message = ?e);
        }
        self
    }

    #[inline]
    #[track_caller]
    fn and_attached_err<A>(self, attachment: A) -> Self
    where
        A: fmt::Debug + Send + Sync + 'static,
    {
        if let Err(e) = &self {
            error!(err = ?e, "{attachment:?}");
        }
        self
    }
}

/// Trait for clearing either the success or error part of a `Result`.
pub trait ClearResult<T, E> {
    /// Clear the error type, converting any error to `()`.
    #[allow(clippy::result_unit_err)]
    fn clear_err(self) -> Result<T, ()>;

    /// Clear the success type, converting any success value to `()`.
    fn clear_ok(self) -> Result<(), E>;
}

impl<T, E> ClearResult<T, E> for Result<T, E> {
    fn clear_err(self) -> Result<T, ()> {
        self.map_err(|_| ())
    }

    fn clear_ok(self) -> Result<(), E> {
        self.map(|_| ())
    }
}

/// Extension trait for `Option<T>` that provides methods to convert `None` into error reports.
///
/// This trait allows you to easily convert `Option` values into `Result<T, Report<NotFound>>`
/// with descriptive error messages.
pub trait OptionReport<T>
where
    Self: Sized,
{
    /// Convert `None` into a `NotFound` error with type information.
    ///
    /// This method transforms an `Option<T>` into a `Result<T, Report<NotFound>>`,
    /// automatically attaching the type information of `T` to provide context
    /// about what was expected but not found.
    ///
    /// # Example
    ///
    /// ```
    /// use bigerror::{OptionReport, NotFound, Report};
    ///
    /// let maybe_value: Option<String> = None;
    /// let result: Result<String, Report<NotFound>> = maybe_value.expect_or();
    /// ```
    fn expect_or(self) -> Result<T, Report<NotFound>>;
    /// Convert `None` into a `NotFound` error with key-value context.
    fn expect_kv<K, V>(self, key: K, value: V) -> Result<T, Report<NotFound>>
    where
        K: Display,
        V: Display;
    /// Convert `None` into a `NotFound` error for a missing field.
    fn expect_field(self, field: &'static str) -> Result<T, Report<NotFound>>;

    /// Convert `None` into a `NotFound` error with key-value context where value is debug-formatted.
    #[inline]
    #[track_caller]
    fn expect_kv_dbg<K, V>(self, key: K, value: V) -> Result<T, Report<NotFound>>
    where
        K: Display,
        V: Debug,
    {
        self.expect_kv(key, Dbg(value))
    }

    /// Convert `None` into a `NotFound` error for a missing indexed item.
    #[inline]
    #[track_caller]
    fn expect_by<K: Display>(self, key: K) -> Result<T, Report<NotFound>> {
        self.expect_kv(Index(key), ty!(T))
    }

    /// Convert `None` into a `NotFound` error with a lazily-computed index key.
    #[deprecated(
        note = "Use `bigerror::OptionReport::expect_with` instead",
        since = "0.12.0"
    )]
    #[inline]
    #[track_caller]
    fn expect_by_fn<F, K>(self, key_fn: F) -> Result<T, Report<NotFound>>
    where
        K: Display,
        F: FnOnce() -> K,
    {
        self.expect_kv(Index(key_fn()), ty!(T))
    }

    /// Convert `None` into a `NotFound` error with a lazily-computed index key.
    #[inline]
    #[track_caller]
    fn expect_with<F, K>(self, key_fn: F) -> Result<T, Report<NotFound>>
    where
        K: Display,
        F: FnOnce() -> K,
    {
        self.expect_kv(Index(key_fn()), ty!(T))
    }

    /// Convert `None` into a `NotFound` That is then propagated to a `ThinContext`
    #[inline]
    #[track_caller]
    fn expect_or_ctx<C>(self) -> Result<T, Report<C>>
    where
        C: ThinContext,
    {
        self.expect_or().into_ctx()
    }
}

impl<T> OptionReport<T> for Option<T> {
    #[inline]
    #[track_caller]
    fn expect_or(self) -> Result<T, Report<NotFound>> {
        // TODO #[track_caller] on closure
        // https://github.com/rust-lang/rust/issues/87417
        // self.ok_or_else(|| Report::new(NotFound))
        match self {
            Some(v) => Ok(v),
            None => Err(NotFound::attach_ty::<T>()),
        }
    }

    #[inline]
    #[track_caller]
    fn expect_kv<K, V>(self, key: K, value: V) -> Result<T, Report<NotFound>>
    where
        K: Display,
        V: Display,
    {
        match self {
            Some(v) => Ok(v),
            None => Err(NotFound::attach_kv(key, value)),
        }
    }

    #[inline]
    #[track_caller]
    fn expect_field(self, field: &'static str) -> Result<T, Report<NotFound>> {
        match self {
            Some(v) => Ok(v),
            None => Err(NotFound::with_field(field)),
        }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! __field {
    ({ } @[$field:ident] ref[$($ref:tt)*] var[$($var:tt)+] fns[$($fn:tt)*]) => {
        // EXIT: return a tuple
        (stringify!($field), $($ref)*$($var)*$($fn)*)
    };
    ({ } @[] ref[$($ref:tt)*] var[$($var:tt)+] fns[$($fn:tt)*]) => {
        // EXIT: return a tuple
        (concat!($(stringify!($var),)+), $($ref)*$($var)*$($fn)*)
    };
    ({ & $($lifetime:lifetime)? $($rest:tt)+ }
        @[] ref[$($ref:tt)*] var[] fns[]) => {
        $crate::__field!(
            { $($rest)+ }
            @[]
            ref[$($ref)* /*add*/ &$($lifetime)?]
            var[]
            fns[]
        )
    };
    ({ * $($rest:tt)+ }
        @[] ref[$($ref:tt)*] var[] fns[]) => {
        $crate::__field!(
            { $($rest)+ }
            @[]
            ref[$($ref)* /*add*/ *]
            var[]
            fns[]
        )
    };
    ({ .%$field:ident $($rest:tt)* }
        @[] ref[$($ref:tt)*] var[$($var:tt)+] fns[$($fn:tt)*]) => {
        $crate::__field!(
            { /*add*/ .$field $($rest)* }
            @[/*add*/ $field]
            ref[$($ref)*]
            var[$($var)+]
            fns[$($fn)*]
        )
    };
    ({ .$method:ident($($arg:expr)? $(, $tail_args:expr)*) $($rest:tt)* }
        @[$($field:ident)?] ref[$($ref:tt)*] var[$($var:tt)*] fns[$($fn:tt)*]) => {
        $crate::__field!(
            { $($rest)* }
            @[$($field)?]
            ref[$($ref)*]
            var[$($var)*]
            fns[$($fn)* /*add*/ .$method($($arg)? $(, $tail_args)*)]
        )
    };
    ({ .$property:ident $($rest:tt)* }
        @[$($field:ident)?] ref[$($ref:tt)*] var[$($var:tt)*] fns[$($fn:tt)*]) => {
        $crate::__field!(
            { $($rest)* }
            @[$($field)?]
            ref[$($ref)*]
            var[$($var)* /*add*/ .$property]
            fns[$($fn)*]
        )
    };
    ({ $var:ident $($rest:tt)* }
        @[] ref[$($ref:tt)*] var[] fns[]) => {
        $crate::__field!(
            { $($rest)* }
            @[]
            ref[$($ref)*]
            var[$var]
            fns[]
        )
    };

    // ENTRYPOINT
    ($($body:tt)+) => {
        $crate::__field!( { $($body)+ } @[] ref[] var[] fns[])
    };
}

/// Converts an `Option` to a `Result`, using the extracted field name for error context.
///
/// This macro extracts field names from expressions and uses them to create descriptive
/// `NotFound` errors when the `Option` is `None`. It works with variable names, struct
/// fields, method calls, and complex expressions.
///
/// # Example
///
/// ```
/// use bigerror::{expect_field, NotFound, Report};
///
/// struct User {
///     email: Option<String>,
/// }
///
/// let user = User { email: None };
/// let result: Result<&String, Report<NotFound>> = expect_field!(user.email.as_ref());
/// // Error will show: field "email" is missing
/// assert!(result.is_err());
/// ```
#[macro_export]
macro_rules! expect_field {
    ($($body:tt)+) => {
        {
            let (__field, __expr)= $crate::__field!($($body)+);
            $crate::OptionReport::expect_field(__expr, __field)
        }
    };
}
#[cfg(test)]
#[derive(Default)]
struct MyStruct {
    my_field: Option<()>,
    _string: String,
}

#[cfg(test)]
impl MyStruct {
    fn __field<T>(_property: &'static str, _t: T) {}
    const fn my_field(&self) -> Option<()> {
        self.my_field
    }
}

#[cfg(test)]
mod test {

    #[cfg(not(feature = "std"))]
    use alloc::boxed::Box;
    #[cfg(not(feature = "std"))]
    use alloc::string::String;

    use crate::attachment::Invalid;

    use super::*;

    #[derive(ThinContext)]
    #[bigerror(crate)]
    pub struct MyError;

    macro_rules! assert_err {
        ($result:expr $(,)?) => {
            let result = $result;
            assert!(result.is_err(), "{:?}", result.unwrap());
            if option_env!("PRINTERR").is_some() {
                crate::init_colour();
                #[cfg(feature = "std")]
                println!("\n{:?}", result.unwrap_err());
            }
        };
        ($result:expr, $($arg:tt)+) => {
            let result = $result;
            assert!(result.is_err(), $($arg)+);
            if option_env!("PRINTERR").is_some() {
                crate::init_colour();
                #[cfg(feature = "std")]
                println!("\n{:?}", result.unwrap_err());
            }
        };
    }

    #[test]
    fn report_as() {
        fn output() -> Result<usize, Report<MyError>> {
            "NaN".parse::<usize>().report_as()
        }

        assert_err!(output());
    }
    #[test]
    fn reportable() {
        fn output() -> Result<usize, Report<MyError>> {
            "NaN".parse::<usize>().map_err(MyError::report)
        }

        assert_err!(output());
    }
    #[test]
    fn box_reportable() {
        fn output() -> Result<usize, Box<dyn core::error::Error + Sync + Send>> {
            Ok("NaN".parse::<usize>().map_err(Box::new)?)
        }

        assert_err!(output().map_err(BoxError::from).change_context(MyError));
    }

    #[test]
    fn convresion_error() {
        fn output() -> Result<usize, Report<ConversionError>> {
            "NaN"
                .parse::<usize>()
                .map_err(ConversionError::from::<&str, usize>)
                .attach(ParseError)
        }

        assert_err!(output().change_context(MyError));
    }

    #[test]
    fn error_in_error_handling() {
        fn output() -> Result<usize, Report<ConversionError>> {
            "NaN"
                .parse::<usize>()
                .map_err(ConversionError::from::<&str, usize>)
                .map_err(|e| match "More NaN".parse::<u32>() {
                    Ok(attachment) => e.attach(attachment),
                    Err(attachment_err) => e
                        .attach(ParseError)
                        .attach_kv("\"More Nan\"", attachment_err),
                })
        }

        assert_err!(output().change_context(MyError));
    }
    #[test]
    fn option_report() {
        assert_err!(None::<()>.expect_or());

        let id: u32 = 0xdead_beef;
        assert_err!(None::<bool>.expect_kv("id", id));
        assert!(Some(true).expect_kv("id", id).unwrap());

        struct OptionField<'a> {
            name: Option<&'a str>,
        }

        let field_none = OptionField { name: None };
        assert_err!(field_none.name.expect_field("name"));

        let field_some = OptionField {
            name: Some("biggy"),
        };
        assert_eq!("biggy", field_some.name.expect_field("name").unwrap());
    }

    #[test]
    fn into_ctx() {
        fn output() -> Result<usize, Report<MyError>> {
            "NaN"
                .parse::<usize>()
                .map_err(|e| ConversionError::from::<&str, usize>(e).into_ctx())
        }

        assert_err!(output());
    }

    #[test]
    fn result_into_ctx() {
        fn output() -> Result<usize, Report<MyError>> {
            "NaN"
                .parse::<usize>()
                .map_err(ConversionError::from::<&str, usize>)
                .change_context(MyError)
                // since we're going from MyError to MyError, we should just attach a
                // new Location to the stack
                .into_ctx()
        }

        assert_err!(output());
    }

    #[test]
    fn attach_ty_status() {
        fn try_even(num: usize) -> Result<(), Report<MyError>> {
            if num % 2 != 0 {
                return Err(InvalidInput::attach_ty_status::<usize>(Invalid).into_ctx());
            }
            Ok(())
        }

        let my_input = try_even(3);
        assert_err!(my_input);
    }

    #[test]
    fn expect_field() {
        let my_struct = MyStruct::default();

        let my_field = expect_field!(my_struct.my_field.as_ref());
        assert_err!(my_field);
        let my_field = expect_field!(my_struct.my_field);
        assert_err!(my_field);
        // from field method
        let my_field = expect_field!(*&my_struct.%my_field());
        assert_err!(my_field);
        let my_field = my_struct.my_field;
        let my_field = expect_field!(my_field.to_owned().to_owned());
        assert_err!(my_field);
    }

    // this is meant to be a compile time test of the `__field!` macro
    fn __field() {
        let my_struct = MyStruct::default();
        let (field, val) = __field!(my_struct._string.as_ref());
        MyStruct::__field::<&str>(field, val);
    }

    #[test]
    fn expectation() {
        let my_struct = MyStruct::default();
        let my_field = my_struct
            .my_field
            .ok_or_else(|| InvalidInput::expected_actual("Some", "None"));

        assert_err!(my_field);
    }

    #[test]
    fn attach_ty_val() {
        fn compare(mine: usize, other: usize) -> Result<(), Report<MyError>> {
            if other != mine {
                bail!(
                    InvalidInput::attach("expected my number!")
                        .attach_ty_val(other)
                        .into_ctx()
                );
            }
            Ok(())
        }

        let my_number = 2;
        let other_number = 3;

        assert_err!(compare(my_number, other_number));
    }

    // should behave the same as `test::attach_ty_val`
    // but displays lazy allocation of attachment
    #[test]
    fn attach_kv_macro() {
        let my_number = 2;
        let other_number = 3;
        fn compare(mine: usize, other: usize) -> Result<(), Report<MyError>> {
            if other != mine {
                return Err(InvalidInput::attach("expected my number!"))
                    .attach_with(|| kv!(ty: other)) // <usize>: 3
                    .into_ctx();
            }
            Ok(())
        }
        assert_err!(compare(my_number, other_number));
    }

    #[test]
    fn expect_by() {
        let arr = ["a", "b"];
        let get_oob = arr.get(2).expect_by(2);
        assert_err!(get_oob);
    }
}
