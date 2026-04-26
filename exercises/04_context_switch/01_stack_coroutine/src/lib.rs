//! # Stackful Coroutine and Context Switch (riscv64)
//!
//! In this exercise, you implement the minimal context switch using inline assembly,
//! which is the core mechanism of OS thread scheduling. This crate is **riscv64 only**;
//! run `cargo test` on riscv64 Linux, or use the repo's normal flow (`./check.sh` / `oscamp`) on x86 with QEMU.
//!
//! ## Key Concepts
//! - **Callee-saved registers**: Save and restore them on switch so the switched-away task can resume correctly later.
//! - **Stack pointer `sp`** and **return address `ra`**: Restore them in the new context; the first time we switch to a task, `ret` jumps to `entry`.
//! - Inline assembly: `core::arch::asm!`
//!
//! ## riscv64 ABI (for this exercise)
//! - Callee-saved: `sp`, `ra`, `s0`–`s11`. The `ret` instruction is `jalr zero, 0(ra)`.
//! - First and second arguments: `a0` (old context), `a1` (new context).

#![cfg(target_arch = "riscv64")]
#![feature(naked_functions)]

use std::arch::naked_asm;

/// Saved register state for a task.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct TaskContext {
    pub sp: u64,
    pub ra: u64,
    pub s0: u64,
    pub s1: u64,
    pub s2: u64,
    pub s3: u64,
    pub s4: u64,
    pub s5: u64,
    pub s6: u64,
    pub s7: u64,
    pub s8: u64,
    pub s9: u64,
    pub s10: u64,
    pub s11: u64,
}

impl TaskContext {
    pub const fn empty() -> Self {
        Self {
            sp: 0, ra: 0,
            s0: 0, s1: 0, s2: 0, s3: 0, s4: 0, s5: 0,
            s6: 0, s7: 0, s8: 0, s9: 0, s10: 0, s11: 0,
        }
    }

    pub fn init(&mut self, stack_top: usize, entry: usize) {
        self.ra = entry as u64;
        self.sp = (stack_top & !0xF) as u64;
    }
}

#[unsafe(naked)]
pub unsafe extern "C" fn switch_context(old: &mut TaskContext, new: &TaskContext) {
    naked_asm!(
        "sd   sp,  0*8(a0)\n",
        "sd   ra,  1*8(a0)\n",
        "sd   s0,  2*8(a0)\n",
        "sd   s1,  3*8(a0)\n",
        "sd   s2,  4*8(a0)\n",
        "sd   s3,  5*8(a0)\n",
        "sd   s4,  6*8(a0)\n",
        "sd   s5,  7*8(a0)\n",
        "sd   s6,  8*8(a0)\n",
        "sd   s7,  9*8(a0)\n",
        "sd   s8, 10*8(a0)\n",
        "sd   s9, 11*8(a0)\n",
        "sd   s10,12*8(a0)\n",
        "sd   s11,13*8(a0)\n",

        "ld   sp,  0*8(a1)\n",
        "ld   ra,  1*8(a1)\n",
        "ld   s0,  2*8(a1)\n",
        "ld   s1,  3*8(a1)\n",
        "ld   s2,  4*8(a1)\n",
        "ld   s3,  5*8(a1)\n",
        "ld   s4,  6*8(a1)\n",
        "ld   s5,  7*8(a1)\n",
        "ld   s6,  8*8(a1)\n",
        "ld   s7,  9*8(a1)\n",
        "ld   s8, 10*8(a1)\n",
        "ld   s9, 11*8(a1)\n",
        "ld   s10,12*8(a1)\n",
        "ld   s11,13*8(a1)\n",

        "li   a0, 0\n",
        "li   a1, 0\n",
        "ret\n",
    )
}

const STACK_SIZE: usize = 1024 * 64;

pub fn alloc_stack() -> (Vec<u8>, usize) {
    let buffer = vec![0u8; STACK_SIZE];
    let stack_top = buffer.as_ptr() as usize + STACK_SIZE;
    (buffer, stack_top & !0xF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    extern "C" fn task_entry() {
        COUNTER.store(42, Ordering::SeqCst);
        loop { std::hint::spin_loop(); }
    }

    #[test]
    fn test_alloc_stack() {
        let (buf, top) = alloc_stack();
        assert_eq!(top, buf.as_ptr() as usize + STACK_SIZE);
        assert!(top % 16 == 0);
    }

    #[test]
    fn test_context_init() {
        let (_, top) = alloc_stack();
        let mut ctx = TaskContext::empty();
        let entry = task_entry as usize;
        ctx.init(top, entry);
        assert_eq!(ctx.ra, entry as u64);
        assert!(ctx.sp != 0);
    }

    #[test]
    fn test_switch_to_task() {
        COUNTER.store(0, Ordering::SeqCst);

        static mut MAIN_CTX: *mut TaskContext = std::ptr::null_mut();
        static mut TASK_CTX: *mut TaskContext = std::ptr::null_mut();

        extern "C" fn cooperative_task() {
            COUNTER.store(99, Ordering::SeqCst);
            unsafe { switch_context(&mut *TASK_CTX, &*MAIN_CTX); }
        }

        let (_, top) = alloc_stack();
        let mut main = TaskContext::empty();
        let mut task = TaskContext::empty();
        task.init(top, cooperative_task as usize);

        unsafe {
            MAIN_CTX = &mut main;
            TASK_CTX = &mut task;
            switch_context(&mut main, &task);
        }

        assert_eq!(COUNTER.load(Ordering::SeqCst), 99);
    }
}