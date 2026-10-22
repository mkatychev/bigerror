# rootcause-types

Errors should be composed of reusable parts to make readability consistent.

## example

What if handling an error also produces an error?


code:

https://github.com/mkatychev/rootcause-types/blob/40e164650bbfe57445853987f7794bdbd2ca18cf/rootcause-types/src/lib.rs#L1219-L1233

command (try it yourself):
```sh
$ cargo test -- test::error_in_error_handling --nocapture
```

output:

<img width="913" alt="image" src="https://github.com/knox-networks/rootcause_types/assets/11223234/bed07095-64ad-4fc4-bd5d-c770f6c4b19c">

## extra

If [`just`](https://just.systems/) is installed: `just printerr-all` to visualize all unit test errors

```sql
-- This error crate is intended to
-- enhance rootcause:
-- https://github.com/rootcause-rs/rootcause
-- rootcause `is_fantastic && does_things.in(|_a| functional(approach))` but
-- `rootcause::Report::context` can make code noisy when used often
--
-- ...so why not auto implement a bunch of `From<Error> for MyError`?
-- and provide fundamental building blocks such as `rootcause_types::NotFound`
-- to express common causes for errors and imply correlation?
CREATE CRATE IF NOT EXISTS
  rootcause_types (
    error BIGERROR NOT NULL,
);
```
