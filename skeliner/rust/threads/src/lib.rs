//! The thread pool skeliner's parallel loops run on.
//!
//! By default that is rayon's global pool: every core the process may use,
//! or `RAYON_NUM_THREADS` when that is set, read once when the first loop
//! starts the pool.  [`set`] picks a pool of the given size instead, until
//! it is called with `None`.  Pools are kept once built, so switching back
//! and forth starts no new threads.  The choice is process-wide.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

/// The chosen pool's size; 0 for rayon's global pool.
static CHOSEN: AtomicUsize = AtomicUsize::new(0);
static POOLS: Mutex<BTreeMap<usize, Arc<ThreadPool>>> = Mutex::new(BTreeMap::new());
/// Whether a loop has run on the global pool, which fixes its size.
static GLOBAL_STARTED: AtomicBool = AtomicBool::new(false);

fn pool(n: usize) -> Option<Arc<ThreadPool>> {
    let pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    pools.get(&n).cloned()
}

/// Run the parallel loops on `n` threads from now on, or on rayon's global
/// pool again with `None`.
pub fn set(n: Option<NonZeroUsize>) -> Result<(), ThreadPoolBuildError> {
    let Some(n) = n.map(NonZeroUsize::get) else {
        CHOSEN.store(0, Ordering::SeqCst);
        return Ok(());
    };
    {
        let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
        if let Entry::Vacant(slot) = pools.entry(n) {
            let built = ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|i| format!("skeliner-{i}"))
                .build()?;
            slot.insert(Arc::new(built));
        }
    }
    CHOSEN.store(n, Ordering::SeqCst);
    Ok(())
}

/// The pool size [`set`] chose, if any.
pub fn chosen() -> Option<usize> {
    match CHOSEN.load(Ordering::SeqCst) {
        0 => None,
        n => Some(n),
    }
}

/// How many threads the next parallel loop runs on.  Asking does not start
/// the global pool, so `RAYON_NUM_THREADS` still applies until a loop does.
pub fn current() -> usize {
    if let Some(n) = chosen() {
        return n;
    }
    if GLOBAL_STARTED.load(Ordering::SeqCst) {
        return rayon::current_num_threads();
    }
    // what rayon's global pool will start with
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, NonZeroUsize::get))
}

/// Run `op`, and every parallel loop in it, on the chosen pool.
pub fn install<R: Send>(op: impl FnOnce() -> R + Send) -> R {
    match chosen().and_then(pool) {
        Some(p) => p.install(op),
        None => {
            GLOBAL_STARTED.store(true, Ordering::SeqCst);
            op()
        }
    }
}

/// Below this many items [`map`] runs on the calling thread: spreading so
/// little costs more than it saves, and it leaves the pool unstarted.
pub const MIN_PARALLEL: usize = 1 << 13;

/// `f` of each of `xs`, in order: on the chosen pool for [`MIN_PARALLEL`]
/// items or more, else on the calling thread.
pub fn map<T: Sync, U: Send>(xs: &[T], f: impl Fn(&T) -> U + Sync + Send) -> Vec<U> {
    if xs.len() < MIN_PARALLEL {
        xs.iter().map(f).collect()
    } else {
        install(|| xs.par_iter().map(f).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker_names() -> Vec<String> {
        install(|| {
            (0..256)
                .into_par_iter()
                .map(|_| std::thread::current().name().unwrap_or("").to_owned())
                .collect()
        })
    }

    #[test]
    fn loops_run_on_the_chosen_pool_until_it_is_unset() {
        let three = NonZeroUsize::new(3);
        set(three).unwrap();
        assert_eq!((chosen(), current()), (Some(3), 3));
        assert_eq!(install(rayon::current_num_threads), 3);
        assert!(worker_names().iter().all(|n| n.starts_with("skeliner-")));

        set(NonZeroUsize::new(2)).unwrap();
        assert_eq!(install(rayon::current_num_threads), 2);
        set(three).unwrap();
        assert_eq!(install(rayon::current_num_threads), 3);
        let kept = POOLS.lock().unwrap().len();

        set(None).unwrap();
        assert_eq!(chosen(), None);
        assert_eq!(
            install(rayon::current_num_threads),
            rayon::current_num_threads()
        );
        assert_eq!(current(), rayon::current_num_threads());
        assert!(worker_names().iter().all(|n| !n.starts_with("skeliner-")));
        assert_eq!(
            POOLS.lock().unwrap().len(),
            kept,
            "a pool is built once per size"
        );
    }
}
