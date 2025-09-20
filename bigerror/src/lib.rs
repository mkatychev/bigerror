//! Enhanced error handling library built on top of [`error-stack`].
//!
//! `bigerror` provides ergonomic error handling with rich context attachments, structured error
//! types, and reduced boilerplate. It extends [`error-stack`] with convenient macros, traits,
//! and pre-defined error contexts for common scenarios.
//!
//! # Key Features
//!
//! - **Zero-cost abstractions** - Built on [`error-stack`] with minimal overhead
//! - **Rich context attachments** - Key-value pairs, field status, type information
//! - **Ergonomic macros** - [`kv!`], [`ty!`], [`expect_field!`] for common patterns
//! - **Pre-defined contexts** - [`NotFound`], [`ParseError`], [`Timeout`], etc.
//! - **No-std support** - Works in embedded and constrained environments
//! - **Automatic conversions** - [`From`] implementations and [`ReportAs`] trait
//!
//! # Quick Start
//!
//! ```rust
//! use bigerror::{ThinContext, Report, kv, expect_field, ParseError, IntoContext};
//!
//! // Define your error type
//! #[derive(bigerror::ThinContext)]
//! struct MyError;
//!
//! fn parse_number(input: &str) -> Result<i32, Report<MyError>> {
//!     // Use context conversion for error handling
//!     let num: i32 = input.parse()
//!         .map_err(|e| ParseError::attach_kv("input", input.to_string()).into_ctx())?;
//!
//!     Ok(num)
//! }
//!
//! // Example with expect_field for optional values
//! fn get_config_value() -> Result<&'static str, Report<MyError>> {
//!     let config = Some("production");
//!     expect_field!(config).map_err(|e| e.into_ctx())
//! }
//! ```
//!
//! # Core Concepts
//!
//! ## Error Contexts
//!
//! Error contexts represent different categories of errors. Use the [`ThinContext`] derive
//! macro to create zero-sized error types:
//!
//! ```rust
//! use bigerror::ThinContext;
//!
//! #[derive(ThinContext)]
//! struct NetworkError;
//!
//! #[derive(ThinContext)]
//! struct ValidationError;
//! ```
//!
//! ## Attachments
//!
//! Attach contextual information to errors using various attachment types:
//!
//! ```rust
//! use bigerror::{ThinContext, kv, ty};
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
//! // Using macros for convenience
//! let username = "alice";
//! let error = MyError::attach(kv!(username)); // "username": "alice"
//! let error = MyError::attach(kv!(ty: username));  // <&str>: "alice"
//! ```
//!
//! ## Conversion and Propagation
//!
//! Use [`ReportAs`] for automatic error conversion with context preservation:
//!
//! ```rust
//! use bigerror::{ReportAs, ThinContext, Report};
//!
//! #[derive(ThinContext)]
//! struct MyError;
//!
//! fn parse_number(s: &str) -> Result<i32, Report<MyError>> {
//!     s.parse().report_as() // Automatically converts ParseIntError
//! }
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
//! - [`IoError`] - I/O operation failures
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
/// # Examples
///
/// ```
/// use bigerror::{ThinContext, Report};
///
/// #[derive(bigerror::ThinContext)]
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
    /// context type. It preserves the complete error chain and adds type information
    /// about the original error.
    ///
    /// # Benefits
    ///
    /// - **Automatic conversion** - No need for manual error wrapping
    /// - **Source chain preservation** - Original error details are maintained
    /// - **Type information** - Includes the original error type name
    /// - **Ergonomic** - Works with `?` operator seamlessly
    ///
    /// # Examples
    ///
    /// ## Basic parsing errors
    ///
    /// ```
    /// use bigerror::{ReportAs, ThinContext, Report};
    ///
    /// #[derive(bigerror::ThinContext)]
    /// struct MyError;
    ///
    /// fn parse_number(s: &str) -> Result<i32, Report<MyError>> {
    ///     s.parse().report_as() // Converts ParseIntError automatically
    /// }
    ///
    /// let result = parse_number("not_a_number");
    /// assert!(result.is_err());
    /// // Error includes original ParseIntError details + type information
    /// ```
    ///
    /// ## File operations
    ///
    /// ```
    /// use bigerror::{ReportAs, ThinContext, Report};
    ///
    /// #[derive(bigerror::ThinContext)]
    /// struct ConfigError;
    ///
    /// fn read_config() -> Result<String, Report<ConfigError>> {
    ///     std::fs::read_to_string("config.txt").report_as()
    /// }
    ///
    /// // If file doesn't exist, converts std::io::Error to Report<ConfigError>
    /// ```
    ///
    /// ## Complex error chains
    ///
    /// ```
    /// use bigerror::{ReportAs, ThinContext, Report};
    ///
    /// #[derive(bigerror::ThinContext)]
    /// struct ProcessingError;
    ///
    /// fn process_data() -> Result<i32, Report<ProcessingError>> {
    ///     let content = std::fs::read_to_string("data.txt").report_as()?;
    ///     let number: i32 = content.trim().parse().report_as()?;
    ///     Ok(number * 2)
    /// }
    ///
    /// // Both I/O errors and parse errors are automatically converted
    /// // while preserving the complete error chain
    /// ```
    ///
    /// ## Comparison with manual conversion
    ///
    /// ```
    /// use bigerror::{ReportAs, ThinContext, Report};
    /// use error_stack::ResultExt;
    ///
    /// #[derive(bigerror::ThinContext)]
    /// struct MyError;
    ///
    /// // With report_as() - concise and automatic
    /// fn parse_easy(s: &str) -> Result<i32, Report<MyError>> {
    ///     s.parse().report_as()
    /// }
    ///
    /// // Manual approach - more verbose
    /// fn parse_manual(s: &str) -> Result<i32, Report<MyError>> {
    ///     s.parse().change_context(MyError)
    /// }
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
    /// If the source and target context types are the same, this will perform
    /// an optimized conversion that preserves the original error structure.
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
    fn into_ctx<C2: ThinContext>(self) -> Result<Self::Ok, Report<C2>>;

    /// Chain operations while converting error context (similar to `Result::and_then`).
    fn and_then_ctx<U, F, C2>(self, op: F) -> Result<U, Report<C2>>
    where
        C2: ThinContext,
        F: FnOnce(Self::Ok) -> Result<U, Report<C2>>;

    /// Map the success value while converting error context (similar to `Result::map`).
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
    #[must_use]
    fn attach_kv<K, V>(self, key: K, value: V) -> Self
    where
        K: Display,
        V: Display;

    /// Attach a key-value pair where the value is debug-formatted.
    #[must_use]
    fn attach_kv_dbg<K, V>(self, key: K, value: V) -> Self
    where
        K: Display,
        V: Debug;

    /// Attach a field with its status.
    #[must_use]
    fn attach_field_status<S>(self, name: &'static str, status: S) -> Self
    where
        S: Display;

    /// Attach a debug-formatted value.
    #[must_use]
    fn attach_dbg<A>(self, value: A) -> Self
    where
        A: Debug;

    /// Attach a type-value pair where the type name is the key.
    #[must_use]
    fn attach_ty_val<A>(self, value: A) -> Self
    where
        Self: Sized,
        A: Display,
    {
        self.attach_kv(ty!(A), value)
    }

    /// Attach a lazily evaluated KeyValue path
    /// Note: this is eagerly evaluated, suggested to use in `.map_err` calls
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
    /// # Examples
    ///
    /// ## Basic usage with Option values
    ///
    /// ```
    /// use bigerror::{OptionReport, NotFound, Report};
    ///
    /// let maybe_value: Option<String> = None;
    /// let result: Result<String, Report<NotFound>> = maybe_value.expect_or();
    /// assert!(result.is_err());
    /// // Error will include type information: "<String> not found"
    /// ```
    ///
    /// ## With Vec and collections
    ///
    /// ```
    /// use bigerror::{OptionReport, NotFound, Report};
    ///
    /// let numbers = vec![1, 2, 3];
    /// let result: Result<&i32, Report<NotFound>> = numbers.get(10).expect_or();
    /// assert!(result.is_err());
    /// // Error will show: "<&i32> not found"
    /// ```
    ///
    /// ## Function return values
    ///
    /// ```
    /// use bigerror::{OptionReport, NotFound, Report};
    ///
    /// fn find_user_by_id(id: u64) -> Result<User, Report<NotFound>> {
    ///     // Simulate database lookup that might return None
    ///     let user: Option<User> = None; // Database lookup result
    ///     user.expect_or()
    /// }
    ///
    /// # struct User { name: String }
    /// let result = find_user_by_id(123);
    /// assert!(result.is_err());
    /// // Error will include: "<User> not found"
    /// ```
    ///
    /// ## Chaining with other error handling
    ///
    /// ```
    /// use bigerror::{OptionReport, IntoContext, ThinContext, Report};
    ///
    /// #[derive(ThinContext)]
    /// struct DatabaseError;
    ///
    /// fn get_config() -> Result<String, Report<DatabaseError>> {
    ///     let config: Option<String> = None;
    ///     config.expect_or().map_err(|e| e.into_ctx())
    /// }
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
    #[inline]
    #[track_caller]
    fn expect_by_fn<F, K>(self, key_fn: F) -> Result<T, Report<NotFound>>
    where
        K: Display,
        F: FnOnce() -> K,
    {
        self.expect_kv(Index(key_fn()), ty!(T))
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
            ref[$($ref)* /*add*/ &$($lifetime)? ]
            var[]
            fns[]
        )
    };
    ({ * $($rest:tt)+ }
        @[] ref[$($ref:tt)*] var[] fns[]) => {
        $crate::__field!(
            { $($rest)+ }
            @[]
            ref[$($ref)* /*add*/ * ]
            var[]
            fns[]
        )
    };
    ({ . % $field:ident $($rest:tt)* }
        @[] ref[$($ref:tt)*] var[$($var:tt)+] fns[$($fn:tt)*]) => {
        $crate::__field!(
            { /*add*/ . $field $($rest)* }
            @[/*add*/ $field]
            ref[$($ref)*]
            var[$($var)+]
            fns[$($fn)*]
        )
    };
    ({. $method:ident($($arg:expr)? $(,$tail_args:expr)* ) $($rest:tt)* }
        @[$($field:ident)?] ref[$($ref:tt)*] var[$($var:tt)*] fns[$($fn:tt)*]) => {
        $crate::__field!(
            { $($rest)* }
            @[$($field)?]
            ref[$($ref)*]
            var[$($var)*]
            fns[$($fn)* /*add*/ . $method($($arg)? $(, $tail_args)*) ]
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
/// # Examples
///
/// ## Basic field extraction
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
/// assert!(result.is_err());
/// // Error will show: field "email" is missing
/// ```
///
/// ## Method calls
///
/// ```
/// use bigerror::{expect_field, NotFound, Report};
///
/// struct Config {
///     token: Option<String>,
/// }
///
/// impl Config {
///     fn get_token(&self) -> Option<&str> {
///         self.token.as_deref()
///     }
/// }
///
/// let config = Config { token: None };
/// let result: Result<&str, Report<NotFound>> = expect_field!(config.%get_token());
/// assert!(result.is_err());
/// // Error will show: field "get_token" is missing
/// ```
///
/// ## Variable extraction
///
/// ```
/// use bigerror::{expect_field, NotFound, Report};
///
/// let session_id: Option<u64> = None;
/// let result: Result<u64, Report<NotFound>> = expect_field!(session_id);
/// assert!(result.is_err());
/// // Error will show: field "session_id" is missing
/// ```
///
/// ## Complex expressions
///
/// ```
/// use bigerror::{expect_field, NotFound, Report};
///
/// let data: Option<Vec<String>> = Some(vec![]);
/// let result: Result<&String, Report<NotFound>> = expect_field!(data.as_ref().and_then(|v| v.first()));
/// assert!(result.is_err());
/// // Error will show: field "data" is missing
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
