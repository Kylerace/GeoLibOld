const std = @import("std");
const ascii = std.ascii;
const Allocator = std.mem.Allocator;
const time = std.time;
const Timer = time.Timer;
const Random = std.Random;

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    //try stdout.print("test");
    @setEvalBranchQuota(10000000);
    const alg = Algebra(5,0,0);
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const allocator = arena.allocator();
    defer _ = arena.deinit();
    const table_format: [] const alg.RowFormat = &.{alg.RowFormat{}};//, alg.RowFormat{.blade_fmt = .binary_no_e, .nonzero_scalars_as_blade = true}};
    const ptr = try alg.get_cayley_table(allocator, &alg.gp_cayley, table_format);
    try stdout.print("Algebra Signature: ({},{},{}):\nGeometric Product:\n{s}\n", .{alg.p, alg.n, alg.z, ptr});
    const ptr1 = try alg.get_cayley_table(allocator, &alg.op_cayley, table_format);
    try stdout.print("Outer Product:\n{s}\n", .{ptr1});
    
    const ptr2 = try alg.get_cayley_table(allocator, &alg.ip_cayley,  table_format);
    try stdout.print("Inner Product:\n{s}\n", .{ptr2});
    const ptr3 = try alg.get_cayley_table(allocator, &alg.rp_cayley, table_format);
    try stdout.print("Regressive Product:\n{s}\n", .{ptr3});

    const NVec = alg.NVec;
    
    var buffer: [alg.basis_len * 8 * 1000]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buffer);
    const fba_alloc = fba.allocator();

    const obj_pool: []NVec = try fba_alloc.alloc(NVec, 10);
    defer fba_alloc.free(obj_pool);
    
    const fst: *NVec = obj_pool[0].zeros();
    fst.terms[0b0000] = 1.2;
    fst.terms[0b0010] = 3.0;
    fst.terms[0b1000] = 1.0;
    fst.terms[0b0110] = 5.1;
    fst.terms[0b1100] = -5.2;
    fst.terms[0b1011] = -4.1;
    const snd: *NVec = obj_pool[1].zeros();
    snd.terms[0b0000] = 2.5;
    snd.terms[0b0001] = 0.5;
    snd.terms[0b0010] = 1.0;
    snd.terms[0b1111] = 0.5;
    snd.terms[0b1001] = 2.3;

    const third: *NVec = obj_pool[2].zeros();
    _ = fst.gp(snd, .ptr, .{.ptr = third});

    try stdout.print("{} gp {} = {}\n", .{fst, snd, third});
    //51: .LBB5_69
    _=third.zeros();
    
    _=fst.gp(snd, .ptr, .{.ptr=third}).add(fst, .ptr, .{.ptr=third});

    try stdout.print("{} gp {} + {} = {}\n", .{fst, snd, fst, third});
    //57: .LBB5_79: same add, same commands but slightly different offsets
    //_=fst.gp_get(snd, .ptr, .{.ptr=third}).add(fst, .ptr, .{.ptr=third});

    //try stdout.print("{} gp_get {} + {} = {}\n", .{fst, snd, fst, third});
    
    const frth: *NVec = obj_pool[3].zeros();
    _=fst.ip(snd, .ptr, .{.ptr=frth});
    try stdout.print("{} ip {} = {}\n", .{fst, snd, frth});
    try stdout.print("mag of {} is {}", .{fst, fst.magnitude()});
    //try stdout.print("{s}", .{alg._op_string});

    //try stdout.print("NVec.Blade: {}", .{NVec.Blade.e2});

    //const whatever = NVec.Blade.@"e2";

    //try stdout.print("std.meta.stringToEnum(NVec.Blade, e2) = {?}", .{std.meta.stringToEnum(NVec.Blade, "e2")});
    //try stdout.print("\nfst.get(.e3) = {d}", .{fst.get(.stack, .e3)});
    //fst.get(.ptr,.e3).* = 2.0;

    //try stdout.print("\nfst.get(.e3) after changing = {d}", .{fst.get(.stack, .e3)});



}

pub fn benchmark(stdout: anytype, allocator: Allocator) void {

    var gp_normal_avg_throughput: f128 = 0.0;
    var gp_get_avg_throughput: f128 = 0.0;
    var gp_better_avg_throughput: f128 = 0.0;
    var gp_rolled_avg_throughput: f128 = 0.0;
    var gp_normal_iters: u128 = 0;
    var gp_get_iters: u128 = 0;
    var gp_better_iters: u128 = 0;
    var gp_rolled_iters: u128 = 0;

    const length_milliseconds = 250;

    const max_p = 5;

    const algs: [max_p]type = comptime blk: {
        var algies: [max_p]type = undefined;
        for(1..max_p+1) |i| {
            algies[i-1] = Algebra(i,0,1);
        }
        break :blk algies;
    };

    const nvecs: [max_p]type = comptime blk: {
        var nvies: [max_p]type = undefined;
        for(1..max_p+1) |i| {
            nvies[i-1] = algs[i-1].NVec;
        }
        break :blk nvies;
    };

    inline for(1..max_p+1) |p| {

        gp_normal_avg_throughput = 0.0;
        gp_get_avg_throughput = 0.0;
        gp_better_avg_throughput = 0.0;
        gp_rolled_avg_throughput = 0.0;

        //const alg2 = algs[p-1];
        const nvec = nvecs[p-1];

        const perf_pool: []nvec = try allocator.alloc(nvec, 1000);
        for(0..100) |_| {
            var timer: Timer = Timer.start() catch unreachable;



            while(timer.read() / 1_000_000 < length_milliseconds) {

                for(0..50) |_| {

                    for(0..perf_pool.len) |j| {
                        const m = &perf_pool[j];
                        _=m.gp_get(&perf_pool[(j + 1) % perf_pool.len], .ptr, .{.ptr=&perf_pool[(j-%1) % perf_pool.len]});
                    }
                }
                gp_get_iters += 1;
            }

            gp_get_avg_throughput = 0.5 * gp_get_avg_throughput + 0.5 * @as(f128, @floatFromInt(gp_get_iters)) * 50.0 * @as(f128, @floatFromInt(perf_pool.len)) / @as(f128, @floatFromInt(length_milliseconds));
            gp_get_iters = 0;
            timer.reset();

            while(timer.read() / 1_000_000 < length_milliseconds) {

                for(0..50) |_| {

                    for(0..perf_pool.len) |j| {
                        const m = &perf_pool[j];
                        _=m.gp(&perf_pool[(j + 1) % perf_pool.len], .ptr, .{.ptr=&perf_pool[(j-%1) % perf_pool.len]} );
                    }
                }
                gp_normal_iters += 1;
            }

            gp_normal_avg_throughput = 0.5 * gp_normal_avg_throughput + 0.5 * @as(f128, @floatFromInt(gp_normal_iters)) * 50.0 * @as(f128, @floatFromInt(perf_pool.len)) / @as(f128, @floatFromInt(length_milliseconds));
            gp_normal_iters = 0;
            timer.reset();


            while(timer.read() / 1_000_000 < length_milliseconds) {

                for(0..50) |_| {

                    for(0..perf_pool.len) |j| {
                        const m = &perf_pool[j];
                        _=m.gp_better(&perf_pool[(j + 1) % perf_pool.len], .ptr, .{.ptr=&perf_pool[(j-%1) % perf_pool.len]});
                    }
                }
                gp_better_iters += 1;
            }

            gp_better_avg_throughput = 0.5 * gp_better_avg_throughput + 0.5 * @as(f128, @floatFromInt(gp_better_iters)) * 50.0 * @as(f128, @floatFromInt(perf_pool.len)) / @as(f128, @floatFromInt(length_milliseconds));
            gp_better_iters = 0;
            timer.reset();

            while(timer.read() / 1_000_000 < length_milliseconds) {

                for(0..50) |_| {

                    for(0..perf_pool.len) |j| {
                        const m = &perf_pool[j];

                        _=m.gp_not_unrolled(&perf_pool[(j + 1) % perf_pool.len], .ptr, .{.ptr=&perf_pool[(j-%1) % perf_pool.len]});
                    }
                }
                gp_rolled_iters += 1;
            }

            gp_rolled_avg_throughput = 0.5 * gp_rolled_avg_throughput + 0.5 * @as(f128, @floatFromInt(gp_rolled_iters)) * 50.0 * @as(f128, @floatFromInt(perf_pool.len)) / @as(f128, @floatFromInt(length_milliseconds));
            gp_rolled_iters = 0;
            timer.reset();


        }

        try stdout.print("\nalgebra({},0,1): gp() got {:.4} avg calls / ms, reordered gp() got {:.4} avg calls / ms, no lookup table gp() got {:.4} avg calls / ms, non inlined gp() got {:.4} avg calls / ms", .{p, gp_normal_avg_throughput, gp_get_avg_throughput, gp_better_avg_throughput, gp_rolled_avg_throughput});
    }
}

pub fn factorial(n: anytype) @TypeOf(n) {
    var result = 1;
    var k = n;
    while(k > 1) {
        result *= k;
        k -= 1;
    }
    //for (2..n+1) |k| {
    //    result *= k;
    //}
    return result;
}

pub fn n_choose_k(n: anytype, k: @TypeOf(n)) @TypeOf(n) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}

/// assumes all elements are distinct
/// also assumes T is an integer type, signed or unsigned
pub fn cycle_sort_ret_writes(T: anytype, slice: []T) usize {
    var cycle_elements: usize = 0;
    var cycles: usize = 0;
    for(0..slice.len) |cycle_start| {
        var item = slice[cycle_start];
        var pos = cycle_start;
        //everything behind us is definitely sorted, some things ahead of us may be sorted
        for(slice[cycle_start+1..slice.len]) |ahead| {
            if(ahead < item) {
                pos += 1;
            }
        }

        if(pos != cycle_start) {
            cycles += 1;
            //cycle_elements += 1;
        }

        //if correct and real pos's are different, then there is a cycle of some length that includes item and the item_at_correct_pos's pos
        //and item_at_correct_pos's_correct_pos and so on until item_at_correct_pos's_correct_pos's..._correct_pos = pos
        //all cycles are disjoint, and the number of writes is the sum of the length of each cycle - number of cycles
        while(pos != cycle_start) {
            //writes += 1;
            pos = cycle_start;
            for(slice[cycle_start+1..slice.len]) |ahead| {
                if(ahead < item) {
                    pos += 1;
                }
            }

            cycle_elements += 1;

            const swapped_item = slice[pos];
            slice[pos] = item;
            item = swapped_item;
        }
    }
    return cycle_elements - cycles;
}

fn count_bits(v: usize) usize {
    var val = v;
    var count: usize = 0;
    //var parity: bool = false;
    while(val != 0) {
        //parity = !parity;
        count += 1;
        val &= val - 1;
    }
    return count;
}

//we want 1 for an odd number of 1's, 0 for an even number of 1's (even parity)
//given that we need to map +1->-1 = 1111... and 0->+1 = 0b000...01
//0b000...01 -> 1111..., 0b000... -> 0b000...01
//NOT: ~0b0...01 -> 0b11...10 + 1 will work here, ~0b000... -> 0b111... +2 w/ overflow will work here
//val = ~even_parity(lhs & rhs) + 1 + even_parity(lhs & rhs) & 1
//so flip_value = ~even_parity(lhs & rhs) += 1 + flip_value & 1

fn count_flips(left: usize, rhs: usize) usize {
    var lhs = left >> 1;
    var flips: usize = 0;
    while(lhs != 0) {
        flips += count_bits(lhs & rhs);
        lhs >>= 1;
    }
    //flips += count_bits((left >> (1 + i)) + rhs)
    return flips;
}
fn flips_value(left: usize, rhs: usize) isize {
    var lhs = left >> 1;
    var val: isize = 1;
    while(lhs != 0) {
        //val = val * (1 - 2 * (count_bits(lhs & rhs) & 1));
        val = (val + @as(isize, @bitCast(count_bits(lhs & rhs) & 1))) & 1;
        lhs >>= 1;
    }
    return 1 - 2 * val;
}

fn flips_value_parity(left: usize, rhs: usize, comptime d: comptime_int) isize {
    var lhs = left >> 1;
    var val: isize = 1;
    while(lhs != 0) {
        //val = val * (1 - 2 * (count_bits(lhs & rhs) & 1));
        val &= @intCast(even_parity(lhs & rhs, d) & 1);
        lhs >>= 1;
    }
    return @as(isize, @intCast(1 - 2 * val));
}

fn even_parity(field: usize, comptime d: comptime_int) usize {
    if(d > 64) {
        @compileError("even_parity() doesnt support field sizes greater than 64 bits");
    }
    if(d == 64) {
        field ^= field >> 32;
    }
    if(d >= 32) {
        field ^= field >> 16;
    }
    if(d >= 16) {
        field ^= field >> 8;
    }
    if(d <= 8) {
        const hex_coeff: usize = 0x0101010101010101;
        const hex_andeff: usize = 0x8040201008040201;
        return (((field * hex_coeff) & hex_andeff) % 0x1FF) & 1;
    }
    field ^= field >> 4;
    field &= 0xf;
    return (0x6996 >> field) & 1;
}

fn Algebra(comptime _p: usize, comptime _n: usize, comptime _z: usize) type {
    const _d: usize = _p + _n + _z;
    const basis_size: usize = std.math.pow(usize, 2, _d);


    const bin_basis: [basis_size]usize = comptime blk: {
        var ret: [basis_size]usize = undefined;
        for(0..basis_size) |blade| {
            ret[blade] = blade;
        }
        break :blk ret;
    };
    //_ = bin_basis;
    //we need the 1blade squaring masks

    //var bin_cayley_table

    //scalar field
    //basis[0] = .{ 1, true, 0, .{0 ** _d} };
    //

    var z_bits = _z;
    var p_bits = _p;
    var n_bits = _n;

    //var curr_bit = 0;
    var zero_mask: usize = 0;
    var pos_mask: usize = 0;
    var neg_mask: usize = 0;
    
    for(0.._d) |b| { // bitfield gen
        if(z_bits > 0) {
            z_bits -= 1;
            zero_mask |= (0b1 << b);
        } else if(p_bits > 0) {
            p_bits -= 1;
            pos_mask |= (0b1 << b);
        } else if(n_bits > 0) {
            n_bits -= 1;
            neg_mask |= (0b1 << b);
        }
    }

    //@compileLog("zero, pos, neg: ", zero_mask, pos_mask, neg_mask, "0b10000:", 0b10000, "count_bits(0b1001):", count_bits(0b1001), "count_flips(e12 = 0b0011, e1 = 0b0001):", count_flips(0b0011, 0b0001));

    const BinBlade = struct{
        value: isize,
        blade: usize
    };

    const ProdTerm = struct {
        mult: f64,
        left_blade: usize,
        right_blade: usize
    };

    @setEvalBranchQuota(1000000);

    var bin_gp_cayley_table: [basis_size][basis_size]BinBlade = .{.{.{.value = 0, .blade = 0}} ** basis_size} ** basis_size;
    var bin_op_cayley_table: [basis_size][basis_size]BinBlade = .{.{.{.value = 0, .blade = 0}} ** basis_size} ** basis_size;
    var bin_ip_cayley_table: [basis_size][basis_size]BinBlade = .{.{.{.value = 0, .blade = 0}} ** basis_size} ** basis_size;
    var bin_rp_cayley_table: [basis_size][basis_size]BinBlade = .{.{.{.value = 0, .blade = 0}} ** basis_size} ** basis_size;

    //array {blade idx = {{lhs blade that produces this, rhs blade that produces this}, ...}, ...}
    var bin_gp_cayley_table_inverse: [basis_size][basis_size]ProdTerm = .{.{.{.mult = 1.0, .left_blade = basis_size + 1, .right_blade = basis_size + 1}} ** basis_size} ** basis_size;
    var bin_op_cayley_table_inverse: [basis_size][basis_size]ProdTerm = .{.{.{.mult = 1.0, .left_blade = basis_size + 1, .right_blade = basis_size + 1}} ** basis_size} ** basis_size;
    var bin_ip_cayley_table_inverse: [basis_size][basis_size]ProdTerm = .{.{.{.mult = 1.0, .left_blade = basis_size + 1, .right_blade = basis_size + 1}} ** basis_size} ** basis_size;
    var bin_rp_cayley_table_inverse: [basis_size][basis_size]ProdTerm = .{.{.{.mult = 1.0, .left_blade = basis_size + 1, .right_blade = basis_size + 1}} ** basis_size} ** basis_size;

    var gp_inv_idx: [basis_size]usize = .{0} ** basis_size;
    var op_inv_idx: [basis_size]usize = .{0} ** basis_size;
    var ip_inv_idx: [basis_size]usize = .{0} ** basis_size;
    var rp_inv_idx: [basis_size]usize = .{0} ** basis_size;

    const pseudoscalar_blade: isize = comptime blk: {
        var val: usize = 0;
        for(0.._d) |i| {
            val |= (1 << i);
        }
        break :blk val;
    };

    for(0..basis_size) |lhs| {
        for(0..basis_size) |rhs| {
            const squares: usize = lhs & rhs;

            const lhsI: isize = lhs ^ pseudoscalar_blade;
            const rhsI: isize = rhs ^ pseudoscalar_blade;
            const dual_squares: isize = lhsI & rhsI;
            if(dual_squares == 0) {

                //const ineg_mask: isize = @bitCast(neg_mask);
                //const lhsI_value: isize = -2 * (((@as(isize, @bitCast(lhs)) & pseudoscalar_blade & ineg_mask) ^ count_flips(lhs, pseudoscalar_blade)) & 1) + 1;
                //const rhsI_value: isize =  -2 * (((@as(isize, @bitCast(rhs)) & pseudoscalar_blade & ineg_mask) ^ count_flips(rhs, pseudoscalar_blade)) & 1) + 1;

                const dual_blade: isize = lhsI ^ rhsI ^ pseudoscalar_blade;
                const dual_value: isize = (-2 * (((lhsI & rhsI & @as(isize, @bitCast(neg_mask))) ^ count_flips(lhsI, rhsI)) & 1) + 1);
                //if(dual_value != dual_value / (lhsI_value * rhsI_value)) {
                //    @compileLog("dual value ", dual_value, " != ", dual_value / (lhsI_value * rhsI_value), " for lhs and rhs: ", lhsI_value, rhsI_value);
                //}

                bin_rp_cayley_table[lhs][rhs] = .{.value = dual_value, .blade = @bitCast(dual_blade)};


                bin_rp_cayley_table_inverse[@bitCast(dual_blade)][rp_inv_idx[@bitCast(dual_blade)]] = .{.mult = bin_rp_cayley_table[lhs][rhs].value, .left_blade = lhs, .right_blade = rhs};
                rp_inv_idx[@bitCast(dual_blade)] += 1;
            }

            if(squares & zero_mask != 0) {
                continue;
            }
            const flips = count_flips(lhs, rhs);

            const negative_powers_mod_2: isize = (((squares & neg_mask) ^ flips) & 1);
            const value: isize = -2 * negative_powers_mod_2 + 1;
            //@bitCast(@as(u8,@truncate((((squares & neg_mask) ^ flips) & 1) << 7)));
            const gp_blade: usize = lhs ^ rhs; //xor = or == true, and not and == true
            //@compileLog("lhs: ", lhs, "rhs: ", rhs, "value:", value, "blade:", blade);
            bin_gp_cayley_table[lhs][rhs] = .{.value = value, .blade = gp_blade};
            bin_gp_cayley_table_inverse[gp_blade][gp_inv_idx[gp_blade]] = .{.mult = value, .left_blade = lhs, .right_blade = rhs};
            gp_inv_idx[gp_blade] += 1;
            //if the two blades have no common non scalar factors then they get turned to 0
            const ul = lhs & gp_blade; //lhs & ((lhs | rhs) & ~(lhs & rhs))
            const ur = rhs & gp_blade;
            bin_ip_cayley_table[lhs][rhs] = .{.value = if(ul == 0 or ur == 0) value else 0, .blade = gp_blade};

            bin_ip_cayley_table_inverse[gp_blade][ip_inv_idx[gp_blade]] = .{.mult = bin_ip_cayley_table[lhs][rhs].value, .left_blade = lhs, .right_blade = rhs};
            ip_inv_idx[gp_blade] += 1;

            if(squares == 0) {
                bin_op_cayley_table[lhs][rhs] = .{.value= if(flips & 1) -1 else 1, .blade = gp_blade};
                bin_op_cayley_table_inverse[gp_blade][op_inv_idx[gp_blade]] = .{.mult = bin_op_cayley_table[lhs][rhs].value, .left_blade = lhs, .right_blade = rhs};
                op_inv_idx[gp_blade] += 1;
            }
        }
    }

    //every element of the alg is a subset of the elements of a full multivector
    //they have a dynamic terms array of d elements alongside a constant array of mappings to
    //the indices of the full multivector,
    //and a static terms array of s elements (alongside a constant array of mappings to
    //the indices of a full multivector) representing the terms the MVecSubset keeps as
    //a constant value that isnt stored in the dynamic array.
    //the length of the dynamic and static term arrays must equal the length of the dynamic
    //array of the full multivector of the algebra, which must equal the basis_size of the algebra

    //for example, i can create a MVecSubset(dynamic grades 1 and 2, index 0 is value 1)
    //which for 3d PGA represents a dynamic array of terms of all grade 1 and 2 basis k-vectors
    //and a static term array where the scalar term is value 1 and all other terms not in
    //the dynamic array are value 0.
    const _bin_gp_cayley_table = bin_gp_cayley_table;
    const _bin_op_cayley_table = bin_op_cayley_table;
    const _bin_ip_cayley_table = bin_ip_cayley_table;
    const _bin_rp_cayley_table = bin_rp_cayley_table;

    const _bin_gp_cayley_table_inv = bin_gp_cayley_table_inverse;
    const _bin_op_cayley_table_inv = bin_op_cayley_table_inverse;
    const _bin_ip_cayley_table_inv = bin_ip_cayley_table_inverse;
    const _bin_rp_cayley_table_inv = bin_rp_cayley_table_inverse;

    const _zero_mask = zero_mask;
    const _pos_mask = pos_mask;
    const _neg_mask = neg_mask;
    const alg = struct {
        pub const Alg = @This();
        pub const p: usize = _p;
        pub const n: usize = _n;
        pub const z: usize = _z;
        pub const d: usize = _d;
        pub const basis: [basis_len]BinBlade = bin_basis;
        pub const basis_len: usize = basis_size;

        pub const mask_zero = _zero_mask;
        pub const mask_pos = _pos_mask;
        pub const mask_neg = _neg_mask;

        pub const KnownSignatures = enum {
            PGAnd,
            CGAnd,
            VGAnd,
            STA,
            STAP,
            None,
        };
        pub const signature_name = blk: {
            if(z == 1 and n == 0 and p > 0) {
                break :blk KnownSignatures.PGAnd;
            }
            if(n == 1 and z == 0 and p > 0) {
                break :blk KnownSignatures.CGAnd;
            }
            if(z == 0 and n == 0 and p > 0) {
                break :blk KnownSignatures.VGAnd;
            }
        };

        pub const gp_cayley: [basis_len][] const BinBlade = blk: {
            var arr: [basis_size][] const BinBlade = undefined;
            for(0..basis_size) |i| {
                arr[i] = _bin_gp_cayley_table[i][0.._bin_gp_cayley_table.len];
            }
            break :blk arr;
        };
        pub const op_cayley: [basis_len][] const BinBlade = blk: {
            var arr: [basis_size][] const BinBlade = undefined;
            for(0..basis_size) |i| {
                arr[i] = _bin_op_cayley_table[i][0.._bin_op_cayley_table.len];
            }
            break :blk arr;
        };
        pub const ip_cayley: [basis_len][] const BinBlade = blk: {
            var arr: [basis_size][] const BinBlade = undefined;
            for(0..basis_size) |i| {
                arr[i] = _bin_ip_cayley_table[i][0.._bin_ip_cayley_table.len];
            }
            break :blk arr;
        };
        pub const rp_cayley: [basis_len][] const BinBlade = blk: {
            var arr: [basis_size][] const BinBlade = undefined;
            for(0..basis_size) |i| {
                arr[i] = _bin_rp_cayley_table[i][0.._bin_rp_cayley_table.len];
            }
            break :blk arr;
        };

        pub const gp_inverse: [basis_len][] const ProdTerm = blk: {
            var arr: [basis_size][] const ProdTerm = undefined;
            for(0..basis_size) |i| {
                arr[i] = _bin_gp_cayley_table_inv[i][0.._bin_gp_cayley_table_inv.len];
            }
            break :blk arr;
        };
        pub const op_inverse: [basis_len][] const ProdTerm = blk: {
            var arr: [basis_size][] const ProdTerm = undefined;
            for(0..basis_size) |i| {
                arr[i] = _bin_op_cayley_table_inv[i][0.._bin_op_cayley_table_inv.len];
            }
            break :blk arr;
        };
        pub const ip_inverse: [basis_len][] const ProdTerm = blk: {
            var arr: [basis_size][] const ProdTerm = undefined;
            for(0..basis_size) |i| {
                arr[i] = _bin_ip_cayley_table_inv[i][0.._bin_ip_cayley_table_inv.len];
            }
            break :blk arr;
        };
        pub const rp_inverse: [basis_len][] const ProdTerm = blk: {
            var arr: [basis_size][] const ProdTerm = undefined;
            for(0..basis_size) |i| {
                arr[i] = _bin_rp_cayley_table_inv[i][0.._bin_rp_cayley_table_inv.len];
            }
            break :blk arr;
        };


        pub const BladeFormats = enum {
            const Us = @This();
            normal,
            binary_blades,
            binary_no_e,
            decimal_no_e,

            pub fn should_print_e(self: Us) bool {
                return switch(self) {
                    .binary_no_e => false,
                    .decimal_no_e => false,
                    else => true
                };
            }

            pub fn base(self: Us) usize {
                return switch(self) {
                    .binary_blades, .binary_no_e => 2,
                    .normal, .decimal_no_e => 10,
                };
            }
        };

        pub const RowFormat = struct { //TODO: allow format matrices
            print_parity: bool = true,
            blade_fmt: BladeFormats = .normal,
            ignore_zeroes: bool = false,
            nonzero_scalars_as_blade: bool = false,
        };


        //fn FormatMatrix(rows: usize, columns: usize) type {
        // 
        //    var ret = struct {
        //        const us = @This();
        //        elements: [rows][columns] ElementFormat,
        // 
        //        fn concat_rows() us {
        // 
        //        }
        //    };
        //}
        // 
        //pub const Alignment = enum {
        //    close, //left hand side for columns up side in rows
        //    center,
        //    far
        //};
        // 
        //pub const TableFormat = struct {
        //    element_formats: [] const RowFormat,
        // 
        //    column_alignment: Alignment = .center,
        //    row_alignment: Alignment = .center,
        // 
        //    //number of extra spaces inserted into every row's string slice
        //    horizontal_padding: usize = 0,
        //    ///number of extra newlines inserted between table rows
        //    vertical_padding: usize = 0,
        // 
        //    horizontal_padding_char: u8 = '|',
        //    vertical_padding_char: u8 = '-'
        // 
        //};

        //pub const TableFormat = []RowFormat;

        fn cayley_printer(element: BinBlade, current_slice: []u8, fmt: RowFormat) void {
            const mid_slice = @as(usize, @as(usize, @intFromFloat(std.math.ceil(@as(f64, @floatFromInt(current_slice.len)) / 2.0))))-1;
            for(0..current_slice.len) |i| {
                current_slice[i] = ' ';
            }

            if(fmt.ignore_zeroes == false) {

                if(element.value == 0) { //0
                    current_slice[mid_slice] = '0';
                    return;
                }
            }


            const is_negative: bool = element.value < 0;
            const starting_factor_label = if(z > 0) 0 else 1;
            const num_labels = count_bits(element.blade);

            if(fmt.nonzero_scalars_as_blade == false and num_labels == 0) { //scalar
                if(is_negative and fmt.print_parity == true) {
                    current_slice[mid_slice] = '-';
                    current_slice[mid_slice + 1] = '1';
                } else {
                    current_slice[mid_slice] = '1';
                }
                return;
            }

            var num_chars: usize = num_labels;
            if(fmt.blade_fmt.base() == 2) {
                num_chars = d;
            }
            if(fmt.blade_fmt.should_print_e()) {
                num_chars += 1;
            }
            if(is_negative and fmt.print_parity == true) {
                num_chars += 1;
            }
            
            var left_margin: usize = 0;
            var right_margin: usize = 0;
            var leftover_margin = current_slice.len - num_chars;

            while(leftover_margin > 0) {
                leftover_margin -= 1;
                left_margin += 1;
                if(leftover_margin > 0) {
                    leftover_margin -= 1;
                    right_margin += 1;
                }
            }

            var i = left_margin;
            if(is_negative == true and fmt.print_parity == true) {
                current_slice[i] = '-';
                i += 1;
            }
            if(fmt.blade_fmt.should_print_e() == true) {

                current_slice[i] = 'e';
                i += 1;
            }

            var current_bit_index: usize = 0;
            if(fmt.blade_fmt.base() == 10) {
                for(0..num_labels) |fac_i| {
                    while(current_bit_index < 10) {
                        const one: usize = 0b1;
                        if(element.blade & (one << @truncate(current_bit_index)) != 0) {
                            break;
                        }
                        current_bit_index += 1;
                    }
                    if(fmt.blade_fmt.base() != 2 and current_bit_index > 9) {
                        current_slice[i + fac_i] = 'X';
                        return;
                    }

                    current_slice[i + fac_i] = @truncate(current_bit_index + starting_factor_label + 48);
                    current_bit_index += 1;
                }
            } else {
                var bit_index = d-1;
                while(bit_index >= 0) {
                    current_slice[i + bit_index] = @truncate(((element.blade >> @truncate((d-1)-bit_index)) & 1) + 48);
                    if(bit_index == 0) {
                        break;
                    }
                    bit_index -= 1;
                }

            }
        }



        fn get_cayley_table(allocator: Allocator, cayley: []const[]const BinBlade, fmt: []const RowFormat) ![]u8 {
            var maximum: usize = 0;
            for(fmt) |row| {
                var elem_size = d;
                if(row.print_parity == true) {
                    elem_size += 1;
                }
                if(row.blade_fmt.should_print_e() == true) {
                    elem_size += 1;
                }
                if(elem_size > maximum) {
                    maximum = elem_size;
                }
            }
            const max_chars_per_index: usize = maximum + 1;
            const ret_string = try allocator.alloc(u8, max_chars_per_index * basis_len * basis_len * fmt.len);
            format_table_string(BinBlade, cayley, ret_string, fmt, maximum, cayley_printer);
            return ret_string;
        }

        fn FormatTableElementPrinter(ElemType: type) type {
            return *const fn(elem: ElemType, string_slice: []u8, fmt: RowFormat) void;
        }

        fn format_table_string(elem_type: type, table: []const []const elem_type, buffer: []u8, fmt: [] const RowFormat, max_chars_per_element: usize, element_printer: FormatTableElementPrinter(elem_type)) void {
            var current_string_idx: usize = 0;
            const max_chars_per_index = max_chars_per_element + 1;
            for(0..table.len * fmt.len) |table_r| {
                for(0..table[table_r % table.len].len) |table_c| {
                    var current_slice = buffer[current_string_idx..current_string_idx + max_chars_per_index];
                    const ending_char: u8 = blk: {
                        if(table_c == table[table_r % table.len].len - 1) {
                            break :blk '\n';
                        }
                        break :blk ' ';
                    };

                    defer {
                        current_slice[current_slice.len - 1] = ending_char;
                        current_string_idx += max_chars_per_index;
                    }

                    element_printer(table[table_r % table.len][table_c], current_slice[0..current_slice.len - 1], fmt[table_r % fmt.len]);
                }
            }
        }

        fn GenBlade() type {

                //we want .fields,
                //var blade = @typeInfo(enum(isize){});
                var fields: [Alg.basis_len]std.builtin.Type.EnumField = undefined;
                var decls = [_]std.builtin.Type.Declaration{};
                //const existing_fields = @typeInfo(Self.ResTypes).Enum.fields;
                //@compileLog( "@typeInfo(Self.ResTypes).Enum.fields) = ", existing_fields, "existing_fields[0].name: ", existing_fields[0].name);
                for(&fields, 0..Alg.basis_len) |*field, i| {
                    var name1: [:0] const u8 = "";
                    var n_i = 0;
                    if(i == 0) {

                        //name1[n_i] = 's';
                        name1 = name1 ++ "s";
                        n_i += 1;
                    } else {
                        //name1[n_i] = 'e';
                        name1 = name1 ++ "e";
                        n_i += 1;

                        for(0..Alg.d+1) |n_j| {

                            const val = (i >> n_j) & 1;
                            if(val != 0) {
                                var char: u8 = '1' + n_j;
                                if(Alg.z > 0) {
                                    char -= 1;
                                }
                                //name1[n_i] = char;
                                name1 = name1 ++ .{char};
                                n_i+=1;
                            }
                        }

                    }
                    

                    //const used_name: [:0]const u8 = name1[0..Alg.d+1]++"";

                    field.* = .{
                        .name = name1,
                        .value = i
                    };
                } 

                const ret = @Type(.{
                    .Enum = .{
                        .tag_type = std.math.IntFittingRange(0, Alg.basis_len-1),
                        .fields = &fields,
                        .decls = &decls,
                        .is_exhaustive = true,
                    }
                });
                //const info = @typeInfo(ret);
                //@compileLog("@typeInfo(reified type) = ", info, "info.Enum = ", info.Enum, "info.Enum.fields = ", info.Enum.fields, "info.Enum.fields[4] (should be e2) = ", info.Enum.fields[4], "name = ", info.Enum.fields[4].name);
                return ret;
        }

        pub const BasisBladeEnum = GenBlade(); //TODO: array?
        pub const BasisBlades: [basis_size]BasisBladeEnum = blk: {
            var ret: [basis_size]BasisBladeEnum = undefined;
            for(0..basis_size) |i| {
                ret[i] = @enumFromInt(i);
            }
            break :blk ret;
        };
        //*const fn(BladeMult, []u8) void

        //this is the base level constructor, we need the dynamic array to be defined by
        // an array the indices of the basis vectors they map to,
        // and the static array to be defined by an array of tuples of the form:
        // (basis index, value they take)
        //fn MVecSubset(comptime T: type, comptime dyn: []const u8, comptime static: [basis_size - dyn.len].{ u8, @TypeOf(T) }) type {

        //}


        pub fn MVecSubset(comptime dyn: [] const BasisBladeEnum) type {

            //Naive (Multi) Vector - full basis_size elements
            const ret = struct {
                pub const Algebra: type = Alg;
                const Self: type = @This();
                pub const ResTypes = enum {
                    alloc_new,
                    ptr, 
                };
                pub const ResSemantics = union {
                    alloc_new: Allocator,
                    ptr: *Self,
                };

                pub const name = [_]u8{"Multivector"};

        
                //pub const unbroken_range: bool = true; ?
                pub const num_terms: usize = Self.Algebra.basis_len;

                ///when converted to MVecSubset this MUST only contain the blades specific to this instance
                pub const Blade: type = Self.Algebra.BasisBladeEnum;
                pub const Blades: [num_terms]Blade = blk: {
                    var ret: [basis_size]BasisBladeEnum = undefined;
                    for(0..basis_size) |i| {
                        ret[i] = @enumFromInt(i);
                    }
                    break :blk ret;
                };

                terms: [dyn.len]f64,

                pub fn get(self: *Self, comptime res_type: enum{stack, ptr}, blade: Blade) switch(res_type) {
                    .stack => f64, .ptr => *f64,
                    } {
                    switch(res_type) {
                        .stack => {
                            const idx: usize = @intFromEnum(blade);
                            return self.terms[idx];
                        },
                        .ptr => {
                            const idx: usize = @intFromEnum(blade);
                            return &self.terms[idx];
                        }, 
                    }
                }

                pub fn zeros(self: *Self) *Self {
                    for(Self.Blades) |b| {
                        self.get(.ptr, b).* = 0;
                    }
                    return self;
                }

                pub fn set_to(self: *Self, source: *Self) *Self {
                    @memcpy(self.terms[0..Self.Algebra.basis_len], source.terms[0..Self.Algebra.basis_len]);
                    return self;
                }

                pub fn format(
                    nvec: NVec,
                    comptime _: []const u8,
                    _: std.fmt.FormatOptions,
                    writer: anytype,
                ) !void {
                    try writer.writeAll("Multivector: {");
                    var has_written_before: bool = false;
                    for(0..Self.Algebra.basis_len) |i| {
                        const term = nvec.terms[i];
                        if(term == 0.0) {
                            continue;
                        }
                        if(has_written_before == true) {
                            _ = try writer.print(" ",.{});
                            if(term > 0.0) {
                                _ = try writer.print("+ ",.{});
                            }
                        }
                        has_written_before = true;

                        var blade_buff: [10]u8 = .{0} ** 10;
                        var chars_needed: usize = count_bits(i);
                        var curr_char_idx: usize = 0;


                        _ = try writer.print("{d:.3}", .{term});

                        if(chars_needed > 0) {
                            blade_buff[curr_char_idx] = 'e';
                            curr_char_idx += 1; //e
                            //var blade_copy = i;
                            var curr_bit_idx: usize = 0b1;
                            var curr_bit_char_num: usize = blk: {
                                if(Self.Algebra.z > 0) {
                                    break :blk 0;
                                }
                                break :blk 1;
                            };
                            while(chars_needed > 0) {
                                if(i & curr_bit_idx == curr_bit_idx) {
                                    chars_needed -= 1;
                                    blade_buff[curr_char_idx] = @truncate(curr_bit_char_num + 48);
                                    curr_char_idx += 1;
                                }
                                curr_bit_idx <<= 1;
                                curr_bit_char_num += 1;
                            }

                            const slice = blade_buff[0..curr_char_idx];
                            _ = try writer.print("{s}",.{slice});
                        }
                        
                    }
                    try writer.writeAll("}");
                }

                pub fn add(lhs: *Self, rhs: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };
                    inline for(0..Self.Algebra.basis_len) |i| {
                        result.terms[i] = lhs.terms[i] + rhs.terms[i];
                    }
                    return result;
                }

                pub fn sub(lhs: *Self, rhs: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };
                    inline for(0..Self.Algebra.basis_len) |i| {
                        result.*.terms[i] = lhs.*.terms[i] - rhs.*.terms[i];
                    }
                    return result;
                }

                pub fn mul_scalar(nvec: *Self, scalar: anytype, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };
                    for(0..Self.Algebra.basis_len) |i| {
                        result.terms[i] = nvec.terms[i] * scalar;
                    }
                    return result;
                }
                

                pub fn gp(left: *Self, right: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };

                    inline for(0..Self.num_terms) |res_blade| {//find every mult pair that produces this blade
                        inner: inline for(0..Self.num_terms) |prod_idx| {
                            const val = comptime blk: {break :blk gp_inverse[res_blade][prod_idx].mult;};
                            const lhs = comptime blk: {break :blk gp_inverse[res_blade][prod_idx].left_blade;};
                            const rhs = comptime blk: {break :blk gp_inverse[res_blade][prod_idx].right_blade;};

                            comptime {
                                if(lhs > Self.num_terms or rhs > Self.num_terms) {
                                    break :inner;
                                }
                            }

                            result.terms[res_blade] += val * (left.terms[lhs] + right.terms[rhs]);
                        }
                    }

                    return result;
                }

                pub fn gp_old(left: *Self, right: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };

                    inline for(0..Self.Algebra.basis_len) |lhs| {
                        inline for(0..Self.Algebra.basis_len) |rhs| {
                            if(Self.Algebra.mask_zero & lhs & rhs == 0) { //should be collapsed at comptime
                                const cayley_elem = comptime blk: { break :blk Self.Algebra.gp_cayley[lhs][rhs];};
                                result.terms[cayley_elem.blade] += cayley_elem.value * left.terms[lhs] * right.terms[rhs];
                            }
                        }
                    }
                    return result;
                }



                pub fn gp_better(left: *Self, right: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };

                    //inline for(0..Self.Algebra.basis_len) |blade| {
                    //    var val: f64 = 0;
                        //2 ({us, 0000} * 2) + 2 ** (d - count_bits(blade))
                        //degenerate indices can be created from 2 ** d total basis blade pairs,
                        //non degenerate indices seem to have half as many? 1 / 2 ** num_degen_bases as many?
                        //we want to find every pair of indices that xor into us
                    //    inline for(blade..Self.Algebra.basis_len) |lhs
                    //}

                    inline for(0..Self.Algebra.basis_len) |lhs| {
                        inline for(0..Self.Algebra.basis_len) |rhs| {
                            if(Self.Algebra.mask_zero & lhs & rhs == 0) { 
                                const squares: isize = lhs & rhs;
                                const flips = count_flips(lhs, rhs);
                                const value: isize = -2 * (((squares & Self.Algebra.mask_neg) ^ @as(isize, @intCast(flips))) & 1) + 1;
                                const gp_blade: usize = lhs ^ rhs;
                                result.terms[gp_blade] += @as(f64, @floatFromInt(value)) * left.terms[lhs] * right.terms[rhs];
                            }
                        }
                    }
                    return result;
                }


                pub fn gp_not_unrolled(left: *Self, right: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };

                    for(0..Self.Algebra.basis_len) |lhs| {
                        for(0..Self.Algebra.basis_len) |rhs| {
                            if(Self.Algebra.mask_zero & lhs & rhs == 0) { //should be collapsed at comptime
                                const cayley_elem = Self.Algebra.gp_cayley[lhs][rhs];
                                result.terms[cayley_elem.blade] += @as(f64, @floatFromInt(cayley_elem.value)) * left.terms[lhs] * right.terms[rhs];
                            }
                        }
                    }
                    return result;
                }

                pub fn op(left: *Self, right: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };

                    inline for(0..Self.Algebra.basis_len) |lhs| {
                        inline for(0..Self.Algebra.basis_len) |rhs| {
                            if(lhs & rhs == 0) {
                                const cayley_elem = comptime blk: { break :blk Self.Algebra.op_cayley[lhs][rhs];};
                                result.terms[cayley_elem.blade] += cayley_elem.value * left.terms[lhs] * right.terms[rhs];
                            }
                        }
                    }

                    return result;
                }


                pub fn ip(left: *Self, right: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };

                    inline for(0..Self.Algebra.basis_len) |lhs| {
                        inline for(0..Self.Algebra.basis_len) |rhs| {
                            if(Self.Algebra.mask_zero & lhs & rhs == 0) { //should be collapsed at comptime
                                const cayley_elem = comptime blk: { break :blk Self.Algebra.ip_cayley[lhs][rhs];};
                                result.terms[cayley_elem.blade] += cayley_elem.value * left.terms[lhs] * right.terms[rhs];
                            }
                        }
                    }

                    return result;
                }
                //e0010 * e1111 = -e1101
                //e0100 * e1111 = e1011
                //e1000 * e1111 = -e0111
                //e1010 * e1111 = (val(e1000 * e1111) + val(e0010 * e1111))e0101
                //blade = ~lhs,
                //value = -2 * (count_flips(lhs, 1111...) & 1) + 1
                //count_flips(0100, 1111):
                //lhs := 0100 >> 1 = 0010
                //lhs != 0
                // flips += count_bits(0010 & 1111) = 0 + 1 = 1
                // lhs = 0010 >> 1 = 0001
                //lhs != 0
                // flips += count_bits(0001 & 1111) = 1 + 1 = 2
                // lhs = 0001 >> 1 = 0000
                //lhs == 0: return 2
                //count_flips(x, 1111...) = sum(count_bits(x >>= 1))
                //if in lhs bit i = 1, then flips += pow(2,i-1) (lsb is index 0)

                pub fn rp(left: *Self, right: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };

                    inline for(0..Self.Algebra.basis_len) |lhs| {
                        inline for(0..Self.Algebra.basis_len) |rhs| {
                            const cayley_elem = comptime blk: { break :blk Self.Algebra.rp_cayley[lhs][rhs];};
                            result.terms[cayley_elem.blade] += cayley_elem.value * left.terms[lhs] * right.terms[rhs];
                        }
                    }

                    return result;
                }

                pub fn grade_project(self: *Self, grade: usize, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = comptime blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };
                    inline for(0..Self.Algebra.basis_len) |i| {
                        if(count_bits(i) == grade) { //theres a better way to do this, the next elem of the same grade is 2^ith indices next
                            result.terms[i] = self.terms[i];
                        } else {
                            result.terms[i] = 0.0;
                        }
                    }
                    return result;
                }


                pub fn reverse(self: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = comptime blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };

                    inline for(0..Self.Algebra.basis_len) |i| {
                        const mult: f64 = comptime blk: {
                            var x: usize = i & ~1;
                            x /= 2;
                            if(x & 1 == 0) {
                                break :blk 1.0;
                            }
                            break :blk -1.0;
                        };
                        result.terms[i] = mult * self.terms[i];
                    }
                    return result;
                }

                pub fn dual(self: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = comptime blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };

                    inline for(0..Self.Algebra.basis_len) |i| {
                        result.terms[i] = self.terms[Self.Algebra.basis_len - 1 - i];
                    }
                    return result;
                }

                pub fn involution(self: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    const result: *Self = comptime blk: {
                        switch(res_type) {
                            .alloc_new => break :blk try res_data.alloc_new.create(Self),
                            .ptr =>  break :blk res_data.ptr
                        }
                    };

                    inline for(0..Self.Algebra.basis_len) |i| {
                        const mult: f64 = comptime blk: {
                            if(i & 1 == 0) {
                                break :blk -1.0;
                            }
                            break :blk 1.0;
                        };
                        result.terms[i] = mult * self.terms[i];
                    }
                    return result;
                }


                //pub fn norm(self: Self) -> Scalar {
                //    let scalar_part = (self * self.Conjugate())[0];
                //
                //    scalar_part.abs().sqrt()
                //} we need gp with reverse

                pub fn magnitude_squared(self: *Self) f64 {
                    @setEvalBranchQuota(100000);
                    var scalar: f64 = 0;
                    inline for(0..Self.Algebra.basis_len) |lhs| {
                        inline for(0..Self.Algebra.basis_len) |rhs| {
                            //we only want to consider the blades that result in a scalar
                            if(Self.Algebra.mask_zero & lhs & rhs == 0 and lhs ^ rhs == 0) { //should be collapsed at comptime
                                const cayley_elem = comptime blk: { break :blk Self.Algebra.gp_cayley[lhs][rhs];};
                                scalar += cayley_elem.value * self.terms[rhs] * self.terms[lhs];
                            }
                        }
                    }

                    return scalar;
                }

                pub fn magnitude(self: *Self) f64 {
                    return @sqrt(@abs(self.magnitude_squared()));
                }

                pub fn normalize(self: *Self, comptime res_type: ResTypes, res_data: ResSemantics) *Self {
                    return self.mul_scalar(1.0 / self.magnitude(), res_type, res_data);
                }

            };
            return ret;
        }

        pub const NVec = MVecSubset(BasisBlades[0..basis_size]);
    };

    return alg;
}
