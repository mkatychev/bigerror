#![cfg_attr(not(feature = "std"), no_std)]
#![expect(deprecated, reason = "We use `Context` to maintain compatibility")]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec};

use error_stack::{IntoReport, fmt::ColorMode};

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
pub use error_stack::{self, Context, Report, ResultExt, bail, ensure, report};

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
    fn attach_lazy<A>(attach_lazy: impl FnOnce() -> A) -> Report<Self>
    where
        A: Display,
    {
        Report::new(Self::VALUE).attach_printable(attach_lazy())
    }

    /// Create an error report with a displayable attachment.
    #[track_caller]
    fn attach<A>(value: A) -> Report<Self>
    where
        A: Display,
    {
        Report::new(Self::VALUE).attach_printable(value)
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
    /// This will wrap the error with additional context including the source
    /// error chain and type information.
    fn report_as<C: ThinContext>(self) -> Result<T, Report<C>>;
}

impl<T, E: Context + core::error::Error> ReportAs<T> for Result<T, E> {
    #[inline]
    #[track_caller]
    fn report_as<C: ThinContext>(self) -> Result<T, Report<C>> {
        // TODO #[track_caller] on closure
        // https://github.com/rust-lang/rust/issues/87417
        // self.map_err(|e| Report::new(C::VALUE).attach_printable(e))
        match self {
            Ok(v) => Ok(v),
            Err(e) => {
                let ty = any::type_name_of_val(&e);
                let mut external_report = Report::new(e).attach_printable(ty);
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
                    external_report = external_report.attach_printable(child_err);
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
                return mem::transmute::<Self, Report<C2>>(self.attach(*Location::caller()));
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
            Err(ctx) => Err(ctx.into_report().change_context(C2::VALUE)),
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
            Err(ctx) => Err(ctx.into_report().change_context(C2::VALUE)),
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
        self.attach_printable(KeyValue(key, value))
    }

    #[inline]
    #[track_caller]
    fn attach_kv_dbg<K, V>(self, key: K, value: V) -> Self
    where
        K: Display,
        V: Debug,
    {
        self.attach_printable(KeyValue::dbg(key, value))
    }

    #[inline]
    #[track_caller]
    fn attach_field_status<S>(self, name: &'static str, status: S) -> Self
    where
        S: Display,
    {
        self.attach_printable(Field::new(name, status))
    }

    #[inline]
    #[track_caller]
    fn attach_dbg<A>(self, value: A) -> Self
    where
        A: Debug,
    {
        self.attach_printable(Dbg(value))
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
            Err(report) => Err(report.attach_printable(KeyValue(key, value))),
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
            Err(report) => Err(report.attach_printable(KeyValue::dbg(key, value))),
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
            Err(report) => Err(report.attach_printable(Field::new(name, status))),
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
            Err(report) => Err(report.attach_printable(Dbg(value))),
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
    // === exits ===
    // handle optional method calls: self.x.as_ref()
    ($fn:path, @[$($rf:tt)*] @[$($pre:expr)+], % $field_method:ident() $(.$method:ident())* ) => {
        $fn($($rf)*$($pre.)+ $field_method() $(.$method())*, stringify!($field_method))
    };
    // handle optional method calls: self.x.as_ref()
    ($fn:path, @[$($rf:tt)*] @[$($pre:expr)+], $field:ident $(.$method:ident())* ) => {
        $fn($($rf)*$($pre.)+ $field $(.$method())*, stringify!($field))
    };
    ($fn:path, @[$($rf:tt)*] @[$body:expr], $(.$method:ident())* ) => {
        $fn($($rf)*$body$(.$method())*, stringify!($body))
    };

    // === much TTs ===
    ($fn:path, @[$($rf:tt)*] @[$($pre:expr)+], $field:ident . $($rest:tt)+) => {
        $crate::__field!($fn, $($rf:tt)* @[$($pre)+ $field], $($rest)+)
    };

    // === entries ===
    ($fn:path | &$body:ident . $($rest:tt)+) => {
        $crate::__field!($fn, @[&] @[$body], $($rest)+)
    };
    ($fn:path | $body:ident . $($rest:tt)+) => {
        $crate::__field!($fn, @[] @[$body], $($rest)+)
    };

    // simple cases
    ($fn:path | &$field:ident) => {
        $fn(&$field, stringify!($field))
    };
    ($fn:path | $field:ident) => {
        $fn($field, stringify!($field))
    };
}

#[macro_export]
macro_rules! expect_field {
    ($($body:tt)+) => {
        $crate::__field!(
         $crate::OptionReport::expect_field |
            $($body)+
        )
    };
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

    #[derive(Default)]
    struct MyStruct {
        my_field: Option<()>,
        _string: String,
    }

    impl MyStruct {
        fn __field<T>(_t: T, _field: &'static str) {}
        const fn my_field(&self) -> Option<()> {
            self.my_field
        }
    }

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
                .attach_printable(ParseError)
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
                    Ok(attachment) => e.attach_printable(attachment),
                    Err(attachment_err) => e
                        .attach_printable(ParseError)
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
        let my_field = expect_field!(my_struct.%my_field());
        assert_err!(my_field);
    }

    // this is meant to be a compile time test of the `__field!` macro
    fn __field() {
        let my_struct = MyStruct::default();
        __field!(MyStruct::__field::<&str> | &my_struct._string);
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
                    .attach_printable_lazy(|| kv!(ty: other)) // <usize>: 3
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
