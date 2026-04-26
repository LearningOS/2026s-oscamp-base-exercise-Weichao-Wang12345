//! # Stackful Coroutine and Context Switch (riscv64)
//!
//! In this exercise, you implement the minimal context switch using inline assembly,
//! which is the core mechanism of OS thread scheduling. This crate is **riscv64 only**;
//! run `cargo test` on riscv64 Linux, or use the repo's normal flow (`./check.sh` / `oscamp`) on x86 with QEMU.
//!
//! ## Key Concepts
//! - **Callee-saved registers**: Save and restore them on switch so the switched-away task can resume correctly later.
//! - **Stack pointer `sp`** and **return address `ra`**: Restore them in the new context; the first time we switch to a task, `ret` jumps to `ra` (the entry point).
//! - Inline assembly: `core::arch::asm!`
//!
//! ## riscv64 ABI (for this exercise)
//! - Callee-saved: `sp`, `ra`, `s0`–`s11`. The `ret` instruction is `jalr zero, 0(ra)`.
//! - First and second arguments: `a0` (old context), `a1` (new context).
#![feature(naked_functions)]
#![no_std]          // ✅ 关键：禁用 std
extern crate alloc;  // ✅ 关键：启用 alloc

use core::arch::naked_asm;       // ✅ 改用 core
use alloc::vec::Vec;             // ✅ 改用 alloc

/// Saved register state for one task (riscv64). Layout must match the offsets used in the asm below:
/// `sp` at 0, `ra` at 8, then `s0`–`s11` at 16, 24, … 104.
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
            sp: 0,
            ra: 0,
            s0: 0,
            s1: 0,
            s2: 0,
            s3: 0,
            s4: 0,
            s5: 0,
            s6: 0,
            s7: 0,
            s8: 0,
            s9: 0,
            s10: 0,
            s11: 0,
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
        "sd   sp,  0*8(a0)",
        "sd   ra,  1*8(a0)",
        "sd   s0,  2*8(a0)",
        "sd   s1,  3*8(a0)",
        "sd   s2,  4*8(a0)",
        "sd   s3,  5*8(a0)",
        "sd   s4,  6*8(a0)",
        "sd   s5,  7*8(a0)",
        "sd   s6,  8*8(a0)",
        "sd   s7,  9*8(a0)",
        "sd   s8, 10*8(a0)",
        "sd   s9, 11*8(a0)",
        "sd   s10,12*8(a0)",
        "sd   s11,13*8(a0)",

        "ld   sp,  0*8(a1)",
        "ld   ra,  1*8(a1)",
        "ld   s0,  2*8(a1)",
        "ld   s1,  3*8(a1)",
        "ld   s2,  4*8(a1)",
        "ld   s3,  5*8(a1)",
        "ld   s4,  6*8(a1)",
        "ld   s5,  7*8(a1)",
        "ld   s6,  8*8(a1)",
        "ld   s7,  9*8(a1)",
        "ld   s8, 10*8(a1)",
        "ld   s9, 11*8(a1)",
        "ld   s10,12*8(a1)",
        "ld   s11,13*8(a1)",

        "li   a0, 0",
        "li   a1, 0",
        "ret",
    )
}

const STACK_SIZE: usize = 1024 * 64;

pub fn alloc_stack() -> (Vec<u8>, usize) {
    let buffer = alloc::vec![0u8; STACK_SIZE];
    let stack_top = buffer.as_ptr() as usize + STACK_SIZE;
    let stack_top_aligned = stack_top & !0xF;
    (buffer, stack_top_aligned)
}

// 测试部分保持不变
#[cfg(test)]
mod tests {
    use super::*;

    use core::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    extern "C" fn task_entry() {
        COUNTER.store(42, Ordering::SeqCst);
        loop {
            core::hint::spin_loop();
        }
    }

    #[test]
    fn test_alloc_stack() {
        let (buf, top) = alloc_stack();
        assert_eq!(top, buf.as_ptr() as usize + STACK_SIZE);
        assert!(top % 16 == 0);
    }

    #[test]
    fn test_context_init() {
        let (buf, top) = alloc_stack();
        let _ = buf;
        let mut ctx = TaskContext::empty();
        let entry = task_entry as *const () as usize;
        ctx.init(top, entry);
        assert_eq!(ctx.ra, entry as u64);
        assert!(ctx.sp != 0);
    }

    #[test]
    fn test_switch_to_task() {
        COUNTER.store(0, Ordering::SeqCst);

        static mut MAIN_CTX_PTR: *mut TaskContext = core::ptr::null_mut();
        static mut TASK_CTX_PTR: *mut TaskContext = core::ptr::null_mut();

        extern "C" fn cooperative_task() {
            COUNTER.store(99, Ordering::SeqCst);
            unsafe {
                switch_context(&mut *TASK_CTX_PTR, &*MAIN_CTX_PTR);
            }
        }

        let (_stack_buf, stack_top) = alloc_stack();
        let mut main_ctx = TaskContext::empty();
        let mut task_ctx = TaskContext::empty();
        task_ctx.init(stack_top, cooperative_task as *const () as usize);

        unsafe {
            MAIN_CTX_PTR = &mut main_ctx;
            TASK_CTX_PTR = &mut task_ctx;
            switch_context(&mut main_ctx, &task_ctx);
        }

        assert_eq!(COUNTER.load(Ordering::SeqCst), 99);
    }
}