//! Custom [`rootcause`] handlers used to replace functionality that the
//! `error-stack` API exposed directly.
//!
//! These handlers are wired into [`rootcause::Report`] via
//! [`new_custom`](rootcause::Report::new_custom) and
//! [`attach_custom`](rootcause::Report::attach_custom).

use core::fmt;

use rootcause::handlers::{
    AttachmentFormattingPlacement, AttachmentFormattingStyle, AttachmentHandler,
    ContextFormattingStyle, ContextHandler, FormattingFunction,
};

/// Re-export of the location attachment and handler shipped with `rootcause`.
pub use rootcause::hooks::builtin_hooks::location::{Location, LocationHandler};

/// Context handler that renders a context together with its
/// [`core::error::Error::source`] chain.
///
/// This replaces the manual source-chain walking that `error-stack` required:
/// the report formatter follows the `source()` chain for us.
#[derive(Debug, Copy, Clone)]
pub struct SourceChain;

impl<C> ContextHandler<C> for SourceChain
where
    C: core::error::Error,
{
    fn source(context: &C) -> Option<&(dyn core::error::Error + 'static)> {
        context.source()
    }

    fn display(context: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(context, f)
    }

    fn debug(context: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(context, f)
    }

    fn preferred_formatting_style(
        _context: &C,
        report_formatting_function: FormattingFunction,
    ) -> ContextFormattingStyle {
        ContextFormattingStyle {
            function: report_formatting_function,
            follow_source: true,
            follow_source_depth: None,
        }
    }
}

/// Attachment handler that hides its attachment entirely.
///
/// This is the closest analogue to `error-stack`'s `attach_opaque`: the value is
/// retained on the report (so it can be inspected programmatically) but never
/// rendered.
#[derive(Debug, Copy, Clone)]
pub struct Opaque;

impl<A> AttachmentHandler<A> for Opaque {
    fn display(_value: &A, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "An object of type {}", core::any::type_name::<A>())
    }

    fn debug(_value: &A, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "An object of type {}", core::any::type_name::<A>())
    }

    fn preferred_formatting_style(
        _value: &A,
        _report_formatting_function: FormattingFunction,
    ) -> AttachmentFormattingStyle {
        AttachmentFormattingStyle {
            placement: AttachmentFormattingPlacement::Hidden,
            function: FormattingFunction::Display,
            priority: 0,
        }
    }
}

/// Attachment handler that always formats its attachment using [`Debug`].
///
/// This is a handler-based equivalent of wrapping a value in a
/// [`Debug`]-formatting newtype.
///
/// [`Debug`]: core::fmt::Debug
#[derive(Debug, Copy, Clone)]
pub struct DebugFmt;

impl<A> AttachmentHandler<A> for DebugFmt
where
    A: fmt::Debug,
{
    fn display(value: &A, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(value, f)
    }

    fn debug(value: &A, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(value, f)
    }

    fn preferred_formatting_style(
        _value: &A,
        _report_formatting_function: FormattingFunction,
    ) -> AttachmentFormattingStyle {
        AttachmentFormattingStyle {
            function: FormattingFunction::Debug,
            ..Default::default()
        }
    }
}
