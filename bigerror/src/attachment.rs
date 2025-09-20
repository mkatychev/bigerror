#[cfg(not(feature = "std"))]
use core::{any, fmt, ops, time::Duration};
#[cfg(feature = "std")]
use std::{any, fmt, ops, time::Duration};

#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
};

use derive_more as dm;
pub use error_stack::{self, Context, Report, ResultExt};

/// Trait alias for types that can be displayed and used in error attachments.
///
/// This trait combines `Display`, `Debug`, `Send`, `Sync`, and `'static` bounds
/// for types that can be attached to error reports.
pub trait Display: fmt::Display + fmt::Debug + Send + Sync + 'static {}

impl<A> Display for A where A: fmt::Display + fmt::Debug + Send + Sync + 'static {}

/// Trait alias for types that can be debug-formatted and used in error attachments.
///
/// This trait combines `Debug`, `Send`, `Sync`, and `'static` bounds for types
/// that can be debug-formatted in error reports.
pub trait Debug: fmt::Debug + Send + Sync + 'static {}

impl<A> Debug for A where A: fmt::Debug + Send + Sync + 'static {}

/// Wrapper for types that only implement `Debug` to make them displayable.
///
/// This wrapper allows debug-only types to be used where `Display` is required
/// by formatting them using their `Debug` implementation.
#[derive(Debug)]
pub struct Dbg<A: Debug>(pub A);

impl<A: Debug> core::error::Error for Dbg<A> {}

impl Dbg<String> {
    /// Create a `Dbg<String>` by debug-formatting any type.
    ///
    /// This is a convenience method for wrapping debug-formatted values in a string.
    pub fn format(attachment: impl fmt::Debug) -> Self {
        Self(format!("{attachment:?}"))
    }
}

impl<A: Debug> fmt::Display for Dbg<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// A simple key-value pair attachment for error reports.
///
/// This type represents a key-value pair that can be attached to error reports
/// for additional context information.
#[derive(Debug, PartialEq, Eq)]
pub struct KeyValue<K, V>(pub K, pub V);

impl<K: fmt::Display, V: fmt::Display> fmt::Display for KeyValue<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.0, self.1)
    }
}

impl<C: Context> core::error::Error for KeyValue<Type, C> {}
impl<C: Context> core::error::Error for KeyValue<&'static str, C> {}

impl<K: Display, V: Debug> KeyValue<K, Dbg<V>> {
    /// Create a key-value pair where the value is debug-formatted.
    ///
    /// This is a convenience method for creating key-value pairs where
    /// the value only implements `Debug` but not `Display`.
    pub const fn dbg(key: K, value: V) -> Self {
        Self(key, Dbg(value))
    }
}

/// Creates a [`KeyValue`] pair for error attachments with flexible key-value syntax.
///
/// This macro provides two forms:
/// - `kv!(ty: value)` - Uses the type of `value` as the key
/// - `kv!(expression)` - Extracts field/variable names from expressions
///
/// # Examples
///
/// ## Type-based key-value pairs
///
/// ```
/// use bigerror::{kv, KeyValue, ty};
///
/// let number = 42;
/// let kv_pair = kv!(ty: number);
/// assert_eq!(kv_pair, KeyValue(ty!(i32), 42));
///
/// // Works with literals too
/// let kv_literal = kv!(ty: "hello");
/// assert_eq!(kv_literal, KeyValue(ty!(&str), "hello"));
/// ```
///
/// ## Field/variable extraction
///
/// ```
/// use bigerror::{kv, KeyValue};
///
/// let username = "alice";
/// let kv_var = kv!(username);
/// assert_eq!(kv_var, KeyValue("username", "alice"));
///
/// struct User { name: String }
/// let user = User { name: "bob".to_string() };
/// let kv_field = kv!(user.name);
/// assert_eq!(kv_field, KeyValue("user.name", "bob".to_string()));
/// ```
///
/// ## Method calls and complex expressions
///
/// ```
/// use bigerror::{kv, KeyValue};
///
/// struct Config { debug: bool }
/// impl Config {
///     fn is_debug(&self) -> bool { self.debug }
/// }
///
/// let config = Config { debug: true };
/// let kv_method = kv!(config.%is_debug());
/// assert_eq!(kv_method, KeyValue("is_debug", true));
/// ```
#[macro_export]
macro_rules! kv {
    (ty: $value: expr) => {
        $crate::KeyValue($crate::Type::of(&$value), $value)
    };
    (type: $value: expr) => {
        $crate::KeyValue($crate::Type::any(&$value), $value)
    };
    ($($body:tt)+) => {
        {
            let (__key, __value)= $crate::__field!($($body)+);
            $crate::KeyValue(__key, __value)
        }
    };
}

/// Represents a field or property with its associated status.
///
/// Field differs from [`KeyValue`] in that the id/key points to a preexisting
/// field, index, or property of a data structure. This is useful for indicating
/// the status of specific fields in validation or processing contexts.
#[derive(Debug)]
pub struct Field<Id, S> {
    /// The identifiable property of a data structure
    /// such as `hash_map["key"]` or a `struct.property`
    id: Id,
    /// The status or state of the field
    status: S,
}

impl<Id: Display, S: Display> fmt::Display for Field<Id, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.id, self.status)
    }
}

impl<Id: Display, S: Display> Field<Id, S> {
    /// Create a new field with the given identifier and status.
    pub const fn new(key: Id, status: S) -> Self {
        Self { id: key, status }
    }
}
/// Wrapper attachment that refers to the type of an object rather than its value.
///
/// This type is used to attach type information to error reports, which is useful
/// for debugging type-related issues or showing what types were involved in an operation.
#[derive(PartialEq, Eq)]
pub struct Type(&'static str);

impl Type {
    /// Create a type attachment for the given type.
    ///
    /// This will be a const fn when `type_name` becomes const fn in stable Rust.
    #[must_use]
    pub fn of<T>() -> Self {
        Self(simple_type_name::<T>())
    }

    /// Create a type attachment for the type of the given value.
    pub fn of_val<T: ?Sized>(_val: &T) -> Self {
        Self(simple_type_name::<T>())
    }
    /// Create a type attachment with a fully qualified URI
    pub fn any<T: ?Sized>(_val: &T) -> Self {
        Self(any::type_name::<T>())
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.0)
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Type").field(&self.0).finish()
    }
}

/// Creates a [`Type`] attachment for the specified type.
///
/// This macro provides a convenient way to create type attachments for error reports.
/// The type attachment shows the type name in error messages, which is useful for
/// debugging type-related issues or showing what types were involved in an operation.
///
/// # Examples
///
/// ## Basic type attachments
///
/// ```
/// use bigerror::{ty, attachment::Type};
///
/// // Create type attachments for built-in types
/// let string_type = ty!(String);
/// let int_type = ty!(i32);
/// let vec_type = ty!(Vec<u8>);
///
/// // They display as <TypeName>
/// assert_eq!(format!("{}", string_type), "<String>");
/// assert_eq!(format!("{}", int_type), "<i32>");
/// ```
///
/// ## Using with error attachments
///
/// ```
/// use bigerror::{ty, ThinContext, NotFound};
///
/// #[derive(bigerror::ThinContext)]
/// struct MyError;
///
/// // Attach type information to errors
/// let error = MyError::attach(ty!(Vec<String>));
/// ```
///
/// ## In key-value pairs
///
/// ```
/// use bigerror::{ty, KeyValue};
///
/// // Use type as a key in key-value attachments
/// let data = vec![1, 2, 3];
/// let kv = KeyValue(ty!(Vec<i32>), data.len());
/// assert_eq!(format!("{}", kv), "<alloc::vec::Vec<i32>>: 3");
/// ```
///
/// ## Custom types
///
/// ```
/// use bigerror::ty;
///
/// struct CustomStruct {
///     field: String,
/// }
///
/// let custom_type = ty!(CustomStruct);
/// // Will show the full module path in the type name
/// ```
#[macro_export]
macro_rules! ty {
    ($type:ty) => {
        $crate::attachment::Type::of::<$type>()
    };
}

/// Status indicator for something that is already present.
///
/// This is commonly used in field status attachments to indicate
/// that a field or value already exists when it shouldn't.
#[derive(Debug, dm::Display)]
#[display("already present")]
pub struct AlreadyPresent;

/// Status indicator for something that is missing.
///
/// This is commonly used in field status attachments to indicate
/// that a required field or value is missing.
#[derive(Debug, dm::Display)]
#[display("missing")]
pub struct Missing;

/// Status indicator for something that is unsupported.
///
/// This is commonly used to indicate that a feature, operation,
/// or value is not supported in the current context.
#[derive(Debug, dm::Display)]
#[display("unsupported")]
pub struct Unsupported;

/// Status indicator for something that is invalid.
///
/// This is commonly used in field status attachments to indicate
/// that a field or value is invalid or malformed.
#[derive(Debug, dm::Display)]
#[display("invalid")]
pub struct Invalid;

/// Attachment that shows expected vs actual values.
///
/// This is useful for validation errors and mismatches where you want
/// to clearly show what was expected versus what was actually received.
#[derive(Debug)]
pub struct Expectation<E, A> {
    /// The expected value
    pub expected: E,
    /// The actual value that was received
    pub actual: A,
}

/// Attachment that shows a conversion from one type to another.
///
/// This is useful for conversion errors to show the source and target types.
#[derive(Debug)]
pub struct FromTo<F, T>(pub F, pub T);

#[allow(dead_code)]
enum Symbol {
    Vertical,
    VerticalRight,
    Horizontal,
    HorizontalLeft,
    HorizontalDown,
    ArrowRight,
    CurveRight,
    Space,
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let utf8 = match self {
            Self::Vertical => "\u{2502}",       // │
            Self::VerticalRight => "\u{251c}",  // ├
            Self::Horizontal => "\u{2500}",     // ─
            Self::HorizontalLeft => "\u{2574}", // ╴
            Self::HorizontalDown => "\u{252c}", // ┬
            Self::ArrowRight => "\u{25b6}",     // ▶
            Self::CurveRight => "\u{2570}",     // ╰
            Self::Space => " ",
        };
        write!(f, "{utf8}")
    }
}

impl<E: Display, A: Display> fmt::Display for Expectation<E, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let curve_right = Symbol::CurveRight;
        let horizontal_left = Symbol::HorizontalLeft;
        let expected = KeyValue("expected", &self.expected);
        let actual = KeyValue("actual", &self.actual);
        // "expected": expected
        // ╰╴"actual": actual
        write!(f, "{expected}\n{curve_right}{horizontal_left}{actual}")
    }
}
impl<F: Display, T: Display> fmt::Display for FromTo<F, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let curve_right = Symbol::CurveRight;
        let horizontal_left = Symbol::HorizontalLeft;
        let from = KeyValue("from", &self.0);
        let to = KeyValue("to", &self.1);
        // "from": from
        // ╰╴"to": to
        write!(f, "{from}\n{curve_right}{horizontal_left}{to}")
    }
}

/// Wrapper for `Duration` that provides human-readable display formatting.
///
/// This wrapper converts duration values into a readable format like "1H30m45s"
/// instead of the default debug representation.
#[derive(Debug)]
pub struct DisplayDuration(pub Duration);
impl fmt::Display for DisplayDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hms_string(self.0))
    }
}

impl From<Duration> for DisplayDuration {
    fn from(duration: Duration) -> Self {
        Self(duration)
    }
}

impl ops::Deref for DisplayDuration {
    type Target = Duration;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Convert a [`Duration`] into a human-readable "0H00m00s" format string.
///
/// This function formats durations in a compact, readable format:
/// - Milliseconds: "123ms"
/// - Seconds only: "00s"
/// - Minutes and seconds: "05m30s"
/// - Hours, minutes, and seconds: "02H15m30s"
/// - Zero duration: "ZERO"
#[must_use]
pub fn hms_string(duration: Duration) -> String {
    if duration.is_zero() {
        return "ZERO".to_string();
    }
    let s = duration.as_secs();
    let ms = duration.subsec_millis();
    // if only milliseconds available
    if s == 0 {
        return format!("{ms}ms");
    }
    // Grab total hours from seconds
    let (h, s) = (s / 3600, s % 3600);
    let (m, s) = (s / 60, s % 60);

    let mut hms = String::new();
    if h != 0 {
        hms += &format!("{h:02}H");
    }
    if m != 0 {
        hms += &format!("{m:02}m");
    }
    hms += &format!("{s:02}s");

    hms
}

/// Extract the simple name of a type, removing module paths.
///
/// This function returns just the type name without the full module path.
/// For generic types like `Option<T>` or `Vec<T>`, it preserves the full
/// generic syntax.
///
/// # Examples
///
/// ```
/// # use bigerror::attachment::simple_type_name;
/// assert_eq!(simple_type_name::<String>(), "String");
/// assert_eq!(simple_type_name::<Option<i32>>(), "core::option::Option<i32>");
/// ```
#[must_use]
pub fn simple_type_name<T: ?Sized>() -> &'static str {
    let full_type = any::type_name::<T>();
    // Option<T>, [T], Vec<T>
    if full_type.contains(['<', '[']) {
        return full_type;
    }
    full_type.rsplit_once("::").map_or(full_type, |t| t.1)
}

/// Wrapper that explicitly indicates a value is being used as an index key.
///
/// This wrapper is used to indicate that the underlying value is being used
/// as an index key for getter methods in collections, such as `HashMap` keys
/// and `Vec` indices. It helps distinguish between regular values and index keys
/// in error messages.
#[derive(Debug, dm::Display)]
#[display("idx [{0}: {}]", simple_type_name::<I>())]
pub struct Index<I: fmt::Display>(pub I);

#[cfg(test)]
mod test {

    use super::*;
    use crate::MyStruct;

    #[test]
    fn kv_macro() {
        let foo = "Foo";

        // foo: "Foo"
        assert_eq!(kv!(foo), KeyValue("foo", "Foo"));
        // <&str>: "Foo"
        assert_eq!(kv!(ty: foo), KeyValue(Type::of_val(&foo), "Foo"));

        let foo = 13;

        // <i32>: 13
        assert_eq!(kv!(ty: foo), KeyValue(Type::of_val(&foo), 13));
        // ensure literal values are handled correctly
        assert_eq!(kv!(ty: 13), KeyValue(Type::of_val(&13), 13));
    }

    #[test]
    fn kv_macro_var() {
        let foo = "Foo";
        let key_value = kv!(foo.to_owned());

        assert_eq!(key_value, KeyValue("foo", String::from(foo)));
    }

    #[test]
    fn kv_macro_struct() {
        let my_struct = MyStruct {
            my_field: None,
            _string: String::from("Value"),
        };

        let key_value = kv!(my_struct.%my_field());
        assert_eq!(key_value, KeyValue("my_field", None));

        let key_value = kv!(my_struct.%_string);
        assert_eq!(key_value, KeyValue("_string", String::from("Value")));

        let key_value = kv!(my_struct.my_field);
        assert_eq!(key_value, KeyValue("my_struct.my_field", None));
    }
}
