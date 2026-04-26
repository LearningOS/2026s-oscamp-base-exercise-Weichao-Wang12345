#![cfg(target_arch = "riscv64")]
// 彻底抑制所有无关警告（测试环境下必要）
#![allow(
    static_mut_refs, 
    invalid_reference_casting, 
    unused_unsafe, 
    stable_features,
    dead_code // 抑制静态变量未使用警告
)]

/// 任务上下文结构体（RISC-V 64位）
/// 内存布局：sp(0) → ra(8) → s0(16) → s1(24) → ... → s11(104)
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct TaskContext {
    pub sp: u64,    // 栈指针
    pub ra: u64,    // 返回地址
    pub s0: u64,    // 被调用者保存寄存器 s0-s11
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
    /// 创建空上下文
    pub const fn empty() -> Self {
        Self {
            sp: 0, ra: 0, s0: 0, s1: 0, s2: 0, s3: 0,
            s4: 0, s5: 0, s6: 0, s7: 0, s8: 0, s9: 0,
            s10: 0, s11: 0
        }
    }

    /// 初始化上下文（栈顶+入口地址）
    pub fn init(&mut self, stack_top: usize, entry: usize) {
        self.sp = stack_top as u64;
        self.ra = entry as u64;
    }
}

/// 全局上下文（协程切换必须的全局状态，标记为used避免dead_code）
#[allow(dead_code)]
static mut MAIN_CTX: TaskContext = TaskContext::empty();
#[allow(dead_code)]
static mut TASK_CTX: TaskContext = TaskContext::empty();

/// 上下文切换核心函数（RISC-V 64位裸函数）
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context(_old: &mut TaskContext, _new: &TaskContext) {
    core::arch::naked_asm!(
        // 保存旧上下文（a0 = old 指针，RISC-V ABI）
        "sd sp,  0(a0)",
        "sd ra,  8(a0)",
        "sd s0, 16(a0)",
        "sd s1, 24(a0)",
        "sd s2, 32(a0)",
        "sd s3, 40(a0)",
        "sd s4, 48(a0)",
        "sd s5, 56(a0)",
        "sd s6, 64(a0)",
        "sd s7, 72(a0)",
        "sd s8, 80(a0)",
        "sd s9, 88(a0)",
        "sd s10,96(a0)",
        "sd s11,104(a0)",

        // 加载新上下文（a1 = new 指针，RISC-V ABI）
        "ld sp,  0(a1)",
        "ld ra,  8(a1)",
        "ld s0, 16(a1)",
        "ld s1, 24(a1)",
        "ld s2, 32(a1)",
        "ld s3, 40(a1)",
        "ld s4, 48(a1)",
        "ld s5, 56(a1)",
        "ld s6, 64(a1)",
        "ld s7, 72(a1)",
        "ld s8, 80(a1)",
        "ld s9, 88(a1)",
        "ld s10,96(a1)",
        "ld s11,104(a1)",

        // 清理参数寄存器，避免指针泄露
        "mv a0, zero",
        "mv a1, zero",

        // 跳转到新上下文入口（ret = jalr zero, 0(ra)）
        "ret",
    );
}

/// 协程栈大小（64KB）
const STACK_SIZE: usize = 1024 * 64;

/// 分配16字节对齐的协程栈（符合RISC-V ABI）
pub fn alloc_stack() -> (Vec<u8>, usize) {
    let stack_buf = vec![0u8; STACK_SIZE + 15];
    let stack_base = stack_buf.as_ptr() as usize;
    let raw_top = stack_base + STACK_SIZE;
    let aligned_top = raw_top & !0xF; // 16字节对齐

    (stack_buf, aligned_top)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::hint;
    use core::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    extern "C" fn task_entry() {
        COUNTER.store(42, Ordering::SeqCst);
        loop { hint::spin_loop(); }
    }

    #[test]
    fn test_alloc_stack() {
        let (buf, top) = alloc_stack();
        let stack_base = buf.as_ptr() as usize;
        let raw_top = stack_base + STACK_SIZE;
        
        assert_eq!(top, raw_top & !0xF);
        assert!(top % 16 == 0);
        assert!(buf.len() >= STACK_SIZE);
    }

    #[test]
    fn test_context_init() {
        let (buf, top) = alloc_stack();
        let _ = buf;
        let mut ctx = TaskContext::empty();
        let entry = task_entry as *const () as usize;
        
        ctx.init(top, entry);
        
        assert_eq!(ctx.ra, entry as u64);
        assert_ne!(ctx.sp, 0);
        assert!((ctx.sp as usize) % 16 == 0);
    }

    #[test]
    fn test_switch_to_task() {
        COUNTER.store(0, Ordering::SeqCst);

        /// 协作式协程：执行后切回主线程
        extern "C" fn cooperative_task() {
            COUNTER.store(99, Ordering::SeqCst);
            // 严格按照Rust 2024提示使用&raw mut/const，转换为匹配的引用
            unsafe {
                let old_ptr = &raw mut TASK_CTX;
                let new_ptr = &raw const MAIN_CTX;
                switch_context(&mut *old_ptr, &*new_ptr);
            }
        }

        // 分配协程栈并初始化上下文
        let (_stack_buf, stack_top) = alloc_stack();
        unsafe {
            // 初始化任务上下文
            TASK_CTX.init(stack_top, cooperative_task as *const () as usize);
            // 使用&raw mut/const消除所有静态mut引用警告
            let main_ptr = &raw mut MAIN_CTX;
            let task_ptr = &raw const TASK_CTX;
            switch_context(&mut *main_ptr, &*task_ptr);
        }

        // 验证协程执行成功
        assert_eq!(COUNTER.load(Ordering::SeqCst), 99);
    }
}