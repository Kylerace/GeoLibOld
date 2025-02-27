const std = @import("std");
const debug = std.debug;
const builtin = @import("builtin");
const meta = std.meta;
const comptimePrint = std.fmt.comptimePrint;

//const mecha = @import("mecha");

//to investigate an alternative structure of library design where "logical" types representing
//an abstract geometric algebra are decoupled from "data / implementation" types that hold runtime data

//TODO: do i want these in the Blade type?
fn count_bits(v: anytype) @TypeOf(v) {
    var val = v;
    var count: @TypeOf(v) = 0;
    //var parity: bool = false;
    while (val != 0) {
        //parity = !parity;
        count += 1;
        val &= val - 1;
    }
    return count;
}

fn count_flips(left: anytype, rhs: anytype) @TypeOf(left) {
    var lhs = left >> 1;
    var flips: @TypeOf(left) = 0;
    while (lhs != 0) {
        flips += count_bits(lhs & rhs);
        lhs >>= 1;
    }
    //flips += count_bits((left >> (1 + i)) + rhs)
    return flips;
}

fn flip_parity(left: anytype, rhs: @TypeOf(left)) @TypeOf(left) {
    return count_flips(left, rhs) & 1;
}

fn flips_permuted_parity(flip_field: anytype, lhs: anytype, rhs: @TypeOf(lhs)) @TypeOf(lhs) {

    return count_flips(lhs,rhs) ^ (flip_field >> lhs) ^ (flip_field >> rhs) ^ (flip_field >> (lhs ^ rhs)) & 1;
}

fn flips_permuted_neg_parity(neg_mask: anytype, flip_field: @TypeOf(neg_mask), lhs: anytype, rhs: @TypeOf(lhs)) @TypeOf(lhs) {
    return (count_bits(neg_mask & (lhs & rhs)) ^ count_flips(lhs,rhs) ^ (flip_field >> lhs) ^ (flip_field >> rhs) ^ (flip_field >> (lhs ^ rhs))) & 1;
}

fn unsigned_to_signed_parity(unsigned_parity: anytype) isize {
    return (-2 * @as(isize, unsigned_parity & 1) + 1);
}


///returns the type of either x if its not a pointer or x's pointed type if it is, and whether it is a ptr or not
pub fn UnwrapPtrType(comptime x: type) struct {type, bool} {
    const x_info = @typeInfo(x);
    const x_is_ptr = x_info == .Pointer;
    const x_type = blk: {
        if(x_is_ptr) {
            break :blk x_info.Pointer.child;
        }
        break :blk x;
    };
    return .{x_type, x_is_ptr};
}

///handles logical operations given an algebraic signature expressed as pos,negative, and zero squaring basis bitfields (must add to 2^n)
/// only needs to know the masks of the algebra, + the included 
//pub fn GAlgebraOps(comptime zero_mask: usize, comptime pos_mask: usize, comptime neg_mask: usize) type {
//const GAlgebraOps = enum {
//    const full_subset: usize = pos_mask | neg_mask | zero_mask;
// 
//    const basis_len: usize = count_bits(full_subset);
//    if(std.math.log2_int(usize, basis_len) != std.math.log2_int_ceil(usize, basis_len)) {
//        @compileError(comptimePrint("non power of 2 number of 1s in mask bitfields passed to GAlgebraOps! p {b} {} n {b} {} z {b} {}", .{pos_mask,pos_mask,neg_mask,neg_mask,zero_mask,zero_mask}));
//    }
//    const d: usize = std.math.log2_int(usize, basis_len);
//    const pseudoscalar: usize = blk: {
//        var val: usize = 0;
//        for(0..d) |i| {
//            val |= (1 << i);
//        }
//        break :blk val;
//    };
// 
//    const ret = struct {
//        const Self = @This();
// 
//        pub const BladeTerm = struct {
//            ///1 where the ith 1 blade is included, 0 elsewhere
//            /// count_bits(blade)th form
//            blade: usize,
//            value: f64
//        };
// 
//    };
//    return ret;
//}

pub fn BladeMult(T: type) type {
    return struct {
        ///1 where the ith 1 blade is included, 0 elsewhere
        /// count_bits(blade)th form
        blade: usize,
        value: T
    };
}

pub fn ProdTerm(T: type) type {
    return struct { 
        mult: T, 
        left_blade: usize, 
        right_blade: usize 
    };
}

pub fn UnaryProdTerm(T: type) type {
    return struct {
        mult: T,
        source: usize
    };
}

///contains all information necessary and/or convenient for GAlgebraOp
const GAlgebraSignature = struct {
    ///num positive-squaring basis 1blades
    p: usize,
    ///num negative-squaring basis 1blades
    n: usize,
    ///num zero-squaring basis 1blades
    z: usize,
    ///num basis 1blades
    d: usize,
    basis_size: usize,

    pseudoscalar: usize,

    ///holds a 1 in bit index i (0 indexed) if that blade squares to a positive number
    pos_mask: usize,
    ///holds a 1 in bit index i (0 indexed) if that blade squares to a negative number
    neg_mask: usize,
    ///holds a 1 in bit index i (0 indexed) if that blade squares to 
    zero_mask: usize,
    /// holds a 1 in the ith bit index (0-indexed) if blade i in our desired basis has a negative parity relative to the "canonical" blade composed of strictly increasing 1blade factors from left to right
    /// for example, in a basis of 1blades {x,y,z}, we can define the pseudoscalar in our basis to be yxz, which is flipped relative to the canonical binary basis of xyz
    /// because the canonical ordering has the 1st 1blade x in the first place, the 2nd 1blade y in the 2nd, etc. so we would have a 1 in index 111 binary 7 decimal of the bitfield (index 0 is the scalar blade)
    /// 10000000 -> I = e213, e132, or e321, 10000000 >> 111 = 1
    /// 76543210 bit indices
    /// 1000 -> blade e12 = binary 11 is flipped (e21)
    /// (this >> a blade) & 1 == 1 iff that blade is flipped, and is 0 otherwise
    flipped_parities: usize,
    /// i dont think this is actually necessary for any functions in GAlgebraOp
    /// but just in case
    dual: bool,
};

pub fn GAlgebraOp(T: type) type {
    return enum {
        pub const Self: type = @This();
        pub const BladeTerm: type = BladeMult(T);

        GeometricProduct,
        OuterProduct,
        InnerProduct,
        RegressiveProduct,
        SandwichProduct,
        ScalarMul,
        //TODO: figure out how to handle ops with arbitrary input and output sizes,
        //  with outputs being possibly nullable
        //  e.g. Add would be a map from (BladeTerm, BladeTerm) -> (BladeTerm, ?BladeTerm)
        //  note: is this yak shaving?
        Add,
        Sub,

        //unary
        Dual,
        Undual,
        Reverse,
        Involution,

        ///values here are the k in n choose k (n = basis_len)
        pub fn n_ary(self: @This()) comptime_int {
            return switch (self) {
                .GeometricProduct => 2,
                .OuterProduct => 2,
                .InnerProduct => 2,
                .RegressiveProduct => 2,
                .SandwichProduct => 2,
                .ScalarMul => 2,
                .Dual => 1,
                .Undual => 1,
                .Reverse => 1,
                .Involution => 1,
                .Add => 2,
                .Sub => 2,
            };
        }

        pub fn Codomain(self: @This()) type {
            return switch(self) {
                .GeometricProduct => BladeTerm,
                .OuterProduct => BladeTerm,
                .InnerProduct => BladeTerm,
                .RegressiveProduct => BladeTerm,
                .SandwichProduct => BladeTerm,
                .ScalarMul => BladeTerm,
                .Dual => BladeTerm,
                .Undual => BladeTerm,
                .Reverse => BladeTerm,
                .Involution => BladeTerm,
                .Add => struct{BladeTerm, ?BladeTerm},
                .Sub => struct{BladeTerm, ?BladeTerm},
            };
        }

        pub fn max_output_size(self: @This()) comptime_int {
            return switch(self) {
                .GeometricProduct => 1,
                .OuterProduct => 1,
                .InnerProduct => 1,
                .RegressiveProduct => 1,
                .SandwichProduct => 1,
                .ScalarMul => 1,
                .Dual => 1,
                .Undual => 1,
                .Reverse => 1,
                .Involution => 1,
                .Add => 2,
                .Sub => 2,
            };
        }

        pub fn BladeOp(self: Self, sig: GAlgebraSignature) type {
            return switch (self) {
                
                .GeometricProduct => struct {
                    pub fn eval(left: BladeTerm, right: BladeTerm) Codomain(.GeometricProduct) {
                        const zero_mask = sig.zero_mask;
                        const neg_mask = sig.neg_mask;
                        const flip_field = sig.flipped_parities;

                        const lhs = left.blade;
                        const rhs = right.blade;
                        const squares = lhs & rhs;
                        const blade: usize = lhs ^ rhs;
                        if (squares & zero_mask != 0) {
                            return BladeTerm{.value = 0, .blade = 0 };
                        }
                        //const neg_powers_mod_2: isize = @intCast(((squares & neg_mask) ^ count_flips(lhs, rhs)) & 1);
                        //const value: isize = -2 * neg_powers_mod_2 + 1;
                        const value: isize = unsigned_to_signed_parity(flips_permuted_neg_parity(neg_mask, flip_field, lhs, rhs));
                        const ret_val = left.value * right.value * @as(f64, @floatFromInt(value));
                        if(@round(ret_val) != ret_val) {
                            @compileLog(comptimePrint("\n\t\tgp lhs val {} * rhs val {} * parity {} = noninteger val {}", .{left.value, right.value, @as(f64, @floatFromInt(value)), ret_val}));
                        }
                        if(@abs(ret_val) > 10.0) {
                            @compileLog(comptimePrint("\n\t------ret_val from .GeometricProduct is greater than 10: {}", ret_val));
                        }
                        //@compileLog(comptimePrint("\nret_val = {}", .{ret_val}));
                        return BladeTerm{ .value = ret_val, .blade = blade };
                    }
                },
                .OuterProduct => struct {
                    pub fn eval(left: BladeTerm, right: BladeTerm) Codomain(.OuterProduct) {
                        const lhs = left.blade;
                        const rhs = right.blade;
                        const squares = lhs & rhs;
                        const blade = lhs ^ rhs;
                        if (squares != 0) {
                            return .{ .value = 0, .blade = blade };
                        }
                        //return .{ .value = if (count_flips(lhs, rhs) & 1) -1 else 1, .blade = blade };
                        return .{.value = unsigned_to_signed_parity(flips_permuted_parity(sig.flipped_parities, lhs, rhs)), .blade = blade};
                    }
                },
                .InnerProduct => struct {
                    pub fn eval(left: BladeTerm, right: BladeTerm) Codomain(.InnerProduct) {
                        const neg_mask: usize = sig.neg_mask;
                        const zero_mask = sig.zero_mask;
                        const flip_field = sig.flipped_parities;

                        const lhs = left.blade;
                        const rhs = right.blade;
                        const squares = lhs & rhs;
                        const blade = lhs ^ rhs;
                        if (squares & zero_mask != 0) {
                            return .{.value = 0, .blade = blade };
                        }
                        const ul = lhs & blade;
                        const ur = rhs & blade;

                        //const neg_powers_mod_2: isize = @intCast(((squares & neg_mask) ^ count_flips(lhs, rhs)) & 1);
                        //const value: isize = -2 * neg_powers_mod_2 + 1;
                        const value: isize = unsigned_to_signed_parity(flips_permuted_neg_parity(neg_mask, flip_field, lhs, rhs));
                        return .{ .value = if (ul == 0 or ur == 0) left.value * right.value * value else 0, .blade = blade };
                    }
                },

                .RegressiveProduct => struct {
                    pub fn eval(left: BladeTerm, right: BladeTerm) Codomain(.RegressiveProduct) {
                        const neg_mask = sig.neg_mask;
                        const flip_field = sig.flipped_parities;
                        const pseudoscalar = sig.pseudoscalar;

                        const lhs = left.blade ^ pseudoscalar;
                        const rhs = right.blade ^ pseudoscalar;
                        const squares = lhs & rhs;
                        const blade = lhs ^ rhs ^ pseudoscalar;
                        if (squares != 0) {
                            return .{ .value = 0, .blade = blade };
                        }
                            
                        //const value = -2 * (((squares & neg_mask) ^ count_flips(lhs, rhs)) & 1) + 1;
                        
                        return .{ .value = unsigned_to_signed_parity(flips_permuted_neg_parity(neg_mask, flip_field, lhs, rhs)), .blade = blade };
                        //const dual = Operation.Dual.BladeOp();
                        //const undual = Operation.Undual.BladeOp();
                        //const outer = Operation.OuterProduct.BladeOp();
                        //return undual.eval(outer.eval(dual.eval(left), dual.eval(right)));
                    }
                },

                //execution is something like first * first * second * coeff
                .SandwichProduct => struct {
                    pub fn eval(left: BladeTerm, right: BladeTerm) Codomain(.SandwichProduct) {
                        //const bread = left.blade;
                        //const meat = right.blade;
                        const gp = GAlgebraOp.GeometricProduct.BladeOp();
                        const reverse = GAlgebraOp.Reverse.BladeOp();
                        return gp.eval(gp.eval(left, right), reverse.eval(left));
                    }
                },


                //TODO: idk how im supporting this if left isnt known at compile time
                .ScalarMul => struct {
                    pub fn eval(left: f64, right: BladeTerm) Codomain(.ScalarMul) {
                        //const bread = left.blade;
                        //const meat = right.blade;
                        return BladeTerm{.value = left * right.value, .blade = right.blade};
                    }
                },

                .Dual => struct {
                    pub fn eval(us: BladeTerm) Codomain(.Dual) {
                        //const squares = us.blade & sig.pseudoscalar;
                        return .{ .value = us.value * unsigned_to_signed_parity(flips_permuted_neg_parity(sig.neg_mask, sig.flipped_parities, us, sig.pseudoscalar)), .blade = us.blade ^ sig.pseudoscalar };
                    }
                },

                .Undual => struct {
                    pub fn eval(us: BladeTerm) Codomain(.Undual) {
                        //var pseudosquarelar = BladeTerm{.value = 1, .blade = pseudoscalar_blade};
                        //const op = Operation.GeometricProduct.BladeOp();
                        //pseudosquarelar = op.eval(pseudosquarelar, pseudosquarelar);
                        //return op.eval(pseudosquarelar, us);
                        const op = @This().Dual.BladeOp();
                        return op.eval(op.eval(op.eval(us)));
                    }
                },

                .Reverse => struct {
                    pub fn eval(us: BladeTerm) Codomain(.Reverse) {
                        const blade: usize = @truncate(us.blade);
                        const one: usize = 0b1;
                        const mult: f64 = comptime blk: {
                            var x: usize = blade & ~one;
                            x >>= 1;
                            if (x & 1 == 0) {
                                break :blk 1.0;
                            }
                            break :blk -1.0;
                        };
                        return BladeTerm{.blade = blade, .value = mult};
                    }
                },

                .Involution => struct {
                    pub fn eval(us: BladeTerm) Codomain(.Involution) {
                        const blade: usize = @truncate(us.blade);
                        return .{.blade = blade, .value = if(blade & 1) -1.0 else 1.0};
                    }
                },

                .Add => struct {
                    pub fn eval(left: BladeTerm, right: BladeTerm) Codomain(.Add) {
                        if(left.blade == right.blade) {
                            return .{BladeTerm{.blade = left.blade, .value = left.value + right.value}, null};
                        }
                        return .{left, right};
                    }
                },
                
                .Sub => struct {
                    pub fn eval(left: BladeTerm, right: BladeTerm) Codomain(.Sub) {
                        if(left.blade == right.blade) {
                            return .{BladeTerm{.blade = left.blade, .value = left.value - right.value}, null};
                        }
                        return .{left, BladeTerm{.blade = right.blade, .value = -right.value}};
                    }
                }
            };
        }

    };
}

const BasisMetric = enum {
    positive,
    negative,
    zero,

    pub fn square(self: @This()) isize {
        return switch(self) {
            .positive => 1,
            .negative => -1,
            .zero => 0
        };
    }
    pub fn from_square(val: isize) @This() {
        if(val < 0) {
            return .negative;
        }
        if(val == 0) {
            return .zero;
        }
        return .positive;
    } 
};

const BladeFloat = struct {
    blade: usize,
    val: f64,
};

///null val = runtime known, nonnull = comptime known
const RuntimeOrComptimeBlade = struct {
    blade: usize,
    val: ?f64,
};

//const pga3_aliases = .{.{name: "Motor", runtime_subset: "s,ix,iy,iz,xy,yz,zx"}, 
//                       .{name: "Point", runtime_subset: "izy,ixz,iyx,xyz", specific_aliases: "izy=px,ixz=py,iyx=pz,xyz=pi"},
//                       .{name: "NormPoint", runtime_subset: "izy,ixz,iyx", comp_subset: "xyz=1", specific_aliases: "izy=px,ixz=py,iyx=pz,xyz=pi"},
//                       .{name: "Line", runtime_subset: "ix,iy,iz,xy,yz,zx"},
//                       .{name: "IdealLine", runtime_subset: "ix,iy,iz"}}
///named subset of a multivector with basis vectors in a specified order and specified runtime and comptime subsets
const GAlgebraAlias = struct {
    const Self: type = @This();

    name: ?[]const u8,
    ///blades, possibly out of order from the canonical ordering
    ordering: []const usize,
    //specific_aliases: ?[]const struct{usize, []const u8},
};

/// for each char in the input at index i, looks for its match in the ordering string at index j and sets 
/// output[i] = j
/// comptime so that i can return an array without an allocator (which wouldnt work at comptime)
pub fn chars_to_binary_ordering_indices(comptime len: usize, comptime input: []const u8, comptime ordering: []const u8) [len]usize {
    var output: [len]usize = undefined;
    if(input.len > ordering.len or input.len != len) {
        debug.panic("non matching lens between len {}, input {s}, and ordering {s}", .{len, input, ordering});
    }
    for(0..input.len) |i| {
        const char: u8 = input[i];
        var found: bool = false;
        for(0..ordering.len) |j| {
            const ordered_char: u8 = ordering[j];
            if(char != ordered_char) {
                continue;
            }
            found = true;
            output[i] = j;
        }
        if(found == false) {
            debug.panic("could not find char {s} from input {s} inside of ordering {s}!", .{&.{char}, input, ordering});
        }
    }
    return output;
}

/// assumes all elements are distinct
/// also assumes T is an integer type, signed or unsigned
/// sorts in ascending order.
/// used to determine the number of flips in a basis blade characterized by a string instead of a bitfield
pub fn cycle_sort_ret_writes(T: anytype, slice: []T) usize {
    var cycle_elements: usize = 0;
    var cycles: usize = 0;
    for (0..slice.len) |cycle_start| {
        var item = slice[cycle_start];
        var pos = cycle_start;
        //everything behind us is definitely sorted, some things ahead of us may be sorted
        for (slice[cycle_start + 1 .. slice.len]) |ahead| {
            if (ahead < item) {
                pos += 1;
            }
        }

        if (pos != cycle_start) {
            cycles += 1;
            //cycle_elements += 1;
        }

        //if correct and real pos's are different, then there is a cycle of some length that includes item and the item_at_correct_pos's pos
        //and item_at_correct_pos's_correct_pos and so on until item_at_correct_pos's_correct_pos's..._correct_pos = pos
        //all cycles are disjoint, and the number of writes is the sum of the length of each cycle - number of cycles
        while (pos != cycle_start) {
            //writes += 1;
            pos = cycle_start;
            for (slice[cycle_start + 1 .. slice.len]) |ahead| {
                if (ahead < item) {
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

///badly named, its actually a basis blade
const BasisVector = struct {
    name: []const u8,
    blade: usize,
    square: BasisMetric,

    pub fn format(
        vector: @This(),
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype
    ) !void {
        _=try writer.print("BasisVector {{{s}, {b}, {c}}}", .{
            vector.name,
            vector.blade,
            switch(vector.square) {
                .positive => '+',
                .negative => '-',
                .zero => '0'
            }
        });
    }

    pub fn flips(comptime self: @This(), comptime ordering: []const u8) isize {
        const name = self.name;
        if(name.len <= 1) {
            return 1;
        }
        var name_indices = chars_to_binary_ordering_indices(name.len, name, ordering);
        return cycle_sort_ret_writes(usize, &name_indices);
    }
};

pub fn BasisVectorMult(T: type) type {
    return struct {
        basis: BasisVector,
        value: T,
    };
}

pub fn BasisVectorMultOpt(T: type) type {
    return struct {
        basis: BasisVector,
        value: ?T,
    };
}

//fn GAlgebraInfo(dual = true, basis = "0i,+x,+y,+z", ordering="", aliases=pga3_aliases)
//holds info that 2 different MVecSubset / impl types use to distinguish between subsets of their own space vs subsets of other spaces
//1. parse 1-form basis, create p,n,z, d, and masks
//2. generate 2^d basis
//  how to deal with reordering and general ops?
//  cayley_table() doesnt need the full array it just needs subsets + and the square masks, 
//  then when we produce the scatter masks we check what actual index the given opr blade belongs to
//  for each multivector subset operand.
//  so we need a "canonical" ordering like the original Algebra impl + a way to convert from that
//  to the correct orderings
//
//3. generate reordered masks
//4. have a function that goes from canonical blade -> reordered index for that blade
//5. have a function that goes from reordered index -> canonical blade
pub fn GAlgebraInfo(comptime _dual: bool, comptime _basis: []const u8, comptime _ordering: ?[]const u8, comptime _aliases: []const GAlgebraAlias) type {
    var _p: usize = 0;
    var _z: usize = 0;
    var _n: usize = 0;
    var _d: usize = 0;

    var _p_mask: usize = 0;
    var _n_mask: usize = 0;
    var _z_mask: usize = 0;

    //ith blade is 1 if we actually use a flipped parity version of that blade as a basis
    var _flipped_parities: usize = 0;

    var parsed_1basis: []const BasisVector = &.{};
    const scalar = BasisVector{.name = "s",.blade = 0,.square = BasisMetric.positive};
    parsed_1basis = parsed_1basis ++ .{scalar};

    const BasisParseStates = enum {
        ///expects a 0,-, or +, if the next character isnt one of these it explodes
        ParseMetric,
        ///expects an ascii character (non ,)
        /// if it finds one, adds that char to the name of curr_basis_blade and moves to ParseNthBasisChar
        /// otherwise, creates a compile error
        ParseFirstBasisChar,
        ///expects either an ascii character (non ,) or a ,
        /// if it finds an ascii letter it adds it to the name of the current basis blade 
        /// if it finds a , it sets parsed_basis to parsed_basis ++ curr_basis_blade and creates a new basis vector to add to
        /// if the for loop ends at any other stage a compile error is thrown
        ParseNthBasisChar,

    };
    var curr_parse_state: BasisParseStates = .ParseMetric;
    var curr_basis_blade: BasisVector = undefined;
    var curr_basis_i: usize = 0;
    for(0.._basis.len+1) |i| {
        switch(curr_parse_state) {
            .ParseMetric => {
                const curr_char = _basis[i];
                if(curr_char == ' ') {
                    continue;
                }
                if(curr_char == '+') {
                    curr_basis_blade.square = .positive;
                    _p += 1;
                    _p_mask |= (1 << curr_basis_i);
                } else if(curr_char == '-') {
                    curr_basis_blade.square = .negative;
                    _n += 1;
                    _n_mask |= (1 << curr_basis_i);
                } else if(curr_char == '0') {
                    curr_basis_blade.square = .zero;
                    _z += 1;
                    _z_mask |= (1 << curr_basis_i);
                } else {
                    @compileError(comptimePrint("bad char found in basis parsing of GAlgebraInfo! basis: {s}", .{_basis}));
                }
                _d += 1;
                curr_basis_i += 1;
                curr_parse_state = .ParseFirstBasisChar;
            },
            .ParseFirstBasisChar => {
                const curr_char = _basis[i];
                if(curr_char == ',') {
                    @compileError(comptimePrint("GAlgebraInfo error: skipped basis vector name (expected non , but got a ,)", .{}));
                }
                //TODO: make sure curr_char is a letter
                //TODO: allow for e_123 basis blades
                curr_basis_blade.name = &.{curr_char};
                curr_basis_blade.blade = (1 << (curr_basis_i - 1));
                curr_parse_state = .ParseNthBasisChar;
            },
            .ParseNthBasisChar => {
                if(i == _basis.len or _basis[i] == ',') {
                    //const _curr_basis_blade = curr_basis_blade;
                    parsed_1basis = parsed_1basis ++ .{curr_basis_blade};
                    curr_basis_blade = undefined;
                    curr_parse_state = .ParseMetric;
                    continue;
                }
                const curr_char = _basis[i];
                if((curr_char < 'A' or curr_char > 'z') or (curr_char > 'Z' and curr_char < 'a')) {
                    @compileError(comptimePrint("non letter char passed into basis string of GAlgebraInfo! _basis: {s}", .{_basis})); 
                }
                curr_basis_blade.name = curr_basis_blade.name ++ .{curr_char};
            }
        }
    }

    const basis_size: usize = std.math.pow(usize, 2, _d);

    var parsed_basis: [basis_size]BasisVector = undefined;
    parsed_basis[0] = parsed_1basis[0];

    @setEvalBranchQuota(1000000);

    for(1..basis_size) |blade| {
        const blade_size: usize = count_bits(blade);
        if(blade_size < 2) {
            for(0.._d) |i| {
                if(blade & (1 << i) == 0) {
                    continue;
                }
                parsed_basis[blade] = parsed_1basis[i+1];
                break;
            }
            continue;
        }

        var blade_name: []const u8 = "";
        var blade_metric = BasisMetric.positive;
        
        for(0.._d) |i| {
            if(blade & (1 << i) == 0) {
                continue;
            }
            blade_name = blade_name ++ .{parsed_1basis[i+1].name[0]};
            const factor_metric: BasisMetric = parsed_1basis[i+1].square;
            blade_metric = BasisMetric.from_square(factor_metric.square() * blade_metric.square());
        }
        blade_metric = BasisMetric.from_square(blade_metric.square() * (-2 * @as(isize, flip_parity(blade, blade)) + 1));
        const new_blade: BasisVector = .{.blade = blade, .name = blade_name, .square = blade_metric};
        parsed_basis[blade] = new_blade;
    }

    //const pseudoscalar_name: []const u8 = parsed_basis[basis_size-1].name;
    //association of [char in _basis] : [index of that char in all blades in the binary blade ordering scheme]
    //also includes scalar basis char 's' as 0
    const binary_ordered_chars: [_d+1]u8 = blk: {
        var ret: [_d+1]u8 = .{'!'} ** (_d + 1);
        for(0..parsed_1basis.len) |i| {
            ret[i] = parsed_1basis[i].name[0];
        }
        break :blk ret;
    };

    // the index of each basis blade is their binary blade rep, but names are flipped if the user flips them
    var _canonical_basis_with_flips: [basis_size]BasisVector = parsed_basis;
    var ordered_basis: [basis_size]BasisVector = parsed_basis;
    //ordering = "s,xy,yz,zx,x,xyz,y,z"
    const OrderingParseStates = enum {
        ParseFirstBasisChar,
        ParseNthBasisChar,
    };
    curr_basis_blade = undefined;
    curr_basis_i = 0;
    var curr_order_parse_state: OrderingParseStates = .ParseFirstBasisChar;
    if(_ordering != null and _ordering.?.len > 0) {
        const ordering = _ordering orelse unreachable;
        //i need to go through the ordering list and parse out every full blade
        //for each one, find the blade matching every character in its name in parsed_basis
        //if there are switched chars, find how many flips that represents and adjust the square

        for(0..ordering.len+1) |i| {
            switch(curr_order_parse_state) {
                //step 1: parse out the next blade, move to step 2 when we encounter an end (, or i == ordering.len)
                //i cannot be at the end here
                .ParseFirstBasisChar => {
                    if(i == ordering.len) {
                        @compileError(comptimePrint("something went wrong with parsing ordering, basis: {any} ordering: {any}", .{_basis, ordering}));
                    }
                    const curr_char = ordering[i];
                    if(curr_char == ' ') {
                        continue;
                    }
                    if(curr_char == ',') {
                        @compileError(comptimePrint("ordering string had a double ,,! ordering: {any}", .{ordering}));
                    }
                    if((curr_char < 'A' or curr_char > 'z') or (curr_char > 'Z' and curr_char < 'a')) {
                        @compileError(comptimePrint("non letter char passed to the order string! ordering: {any}", ordering));
                    }

                    curr_basis_blade = .{.blade = undefined, .name = "", .square = .positive};
                    curr_basis_blade.name = curr_basis_blade.name ++ .{ordering[i]};
                    //at this point it is legal for the current name to end, when it does (parsenth will detect this)
                    //we will find out what blade it belongs to
                    curr_order_parse_state = .ParseNthBasisChar;
                },
                .ParseNthBasisChar => {
                    if(i == ordering.len or ordering[i] == ',') {
                        //we have the full name, now find the blade, and how many flips our name has rel. to the canonical name
                        var blade: usize = 0;
                        const name: []const u8 = curr_basis_blade.name;
                        if(name.len == 0) {
                            @compileError(comptimePrint("empty name found when parsing ordering string! ordering: {any}", .{ordering}));
                        }

                        var name_indices: [name.len]usize = chars_to_binary_ordering_indices(name.len, name, &binary_ordered_chars);
                        //create the blade
                        for(0..name.len) |name_i| {
                            const blade_idx: usize = name_indices[name_i];
                            if(blade_idx == 0) {
                                if(name.len > 1) {
                                    @compileError(comptimePrint("somehow scalar char s got into the ordering string for some blade, name: {s} ordering {s}", .{name, ordering}));
                                }
                                break;
                            }
                            blade |= (1 << (blade_idx - 1)); //s is idx 0
                        }
                        const flips: usize = cycle_sort_ret_writes(usize, &name_indices);
                        const parity: isize = unsigned_to_signed_parity(flips & 1);
                        if(parity < 0) {
                            //const old_flipped = _flipped_parities;
                            _flipped_parities |= (1 << blade);
                            //@compileLog(comptimePrint("flipped blade {s} found to be binary blade {b} with {} flips, _flipped_parities {b} -> {b}, ordering {s}", .{name, blade, flips, old_flipped, _flipped_parities, ordering}));
                        }
                        curr_basis_blade.blade = blade;
                        curr_basis_blade.square = parsed_basis[blade].square;
                        
                        curr_order_parse_state = .ParseFirstBasisChar;

                        ordered_basis[curr_basis_i] = curr_basis_blade;
                        _canonical_basis_with_flips[blade] = curr_basis_blade;
                        curr_basis_i += 1;

                    } else { //parse the next character of the name
                        const curr_char: u8 = ordering[i];
                        if((curr_char < 'A' or curr_char > 'z') or (curr_char > 'Z' and curr_char < 'a')) {
                            @compileError(comptimePrint("non letter char passed to the order string! ordering: {any}", ordering));
                        }
                        if(curr_char == 's' and curr_basis_blade.name.len != 0) {
                            @compileError(comptimePrint("s added to nonzero len blade name! ordering: {s}", .{ordering}));
                        }
                        var in_1basis: bool = false;
                        for(0..parsed_1basis.len) |j| {
                            if(parsed_1basis[j].name[0] == curr_char) {
                                in_1basis = true;
                                break;
                            }
                        }
                        if(in_1basis == false) {//TODO: add to other state
                            @compileError(comptimePrint("char in ordering string blade not in 1 basis! char {s} ordering {s} parsed_1basis: {any}", .{&.{curr_char}, ordering, parsed_1basis}));
                        }
                        curr_basis_blade.name = curr_basis_blade.name ++ .{curr_char};
                    }
                }
            }
        }
        if(curr_basis_i != basis_size) {
            @compileError(comptimePrint("non null ordering string doesnt include all basis elements! ordering {s}, number parsed: {}, ordered_basis: {any}", .{ordering, curr_basis_i, ordered_basis[0..curr_basis_i]}));
        }

    }

    //@compileLog(comptimePrint("_basis: {s}", .{_basis}));
    //@compileLog(comptimePrint("parsed 1 basis: {s}", .{parsed_1basis}));
    //@compileLog(comptimePrint("parsed basis: {s}", .{parsed_basis}));

    //if(_ordering) |ordering| {
    //    @compileLog(comptimePrint("_ordering {s}", .{ordering}));
    //} else {
    //    @compileLog(comptimePrint("_ordering: {?}", .{_ordering}));
    //}
    //@compileLog(comptimePrint("ordered basis: {any}", .{ordered_basis}));
    //@compileLog(comptimePrint("canonical basis with flips: {any}", .{_canonical_basis_with_flips}));

    const sig: GAlgebraSignature = .{
        .p = _p,
        .n = _n,
        .z = _z,
        .d = _d,
        .basis_size = basis_size,
        .pseudoscalar = basis_size - 1,
        .pos_mask = _p_mask,
        .neg_mask = _n_mask,
        .zero_mask = _z_mask,
        .flipped_parities = _flipped_parities,
        .dual = _dual,
    };

    const _parsed_1basis = parsed_1basis;
    const _parsed_basis = parsed_basis;
    const _ordered_basis = ordered_basis;
    const __canonical_basis_with_flips = _canonical_basis_with_flips;
    const ret = struct {
        const Self = @This();
        /// so that users of this can recreate it in type functions using us as a parameter without 
        /// passing in every arg manually or using us as just being type as : type
        /// so that the lang server can see every field in the struct
        pub const GAlgebraInfoArgs = struct {
            pub const __dual: bool = _dual;
            pub const __basis: []const u8 = _basis;
            pub const __ordering: ?[]const u8 = _ordering;
            pub const __aliases: []const GAlgebraAlias = _aliases;
        };

        ///i wish zig had better ways of constraining types to those returned by a type fn
        pub const IsGAlgebraInfo: bool = true;

        pub const signature: GAlgebraSignature = sig;
        pub const dual = _dual;
        pub const basis_1_blades = _parsed_1basis;
        /// canonical binary ordering, blades cannot be flipped
        /// TODO: do i even want this one
        pub const canonical_basis = _parsed_basis;
        /// custom ordering and blades can be flipped
        pub const ordered_basis_blades = _ordered_basis;
        /// binary ordering but blades can be flipped
        pub const canonical_basis_with_flips = __canonical_basis_with_flips;
        pub const aliases = _aliases;

        //cayley space of n-ary blade function = 
        //  operand blades -> result blade: rank n Tensor(BladeTerm, &.{basis_size, basis_size,...}),
        //  result blade mults -> rank 1 Tensor(T, &.{basis_size}),
        //  result blade -> operand blades: rank n Tensor(usize, )
        //BladeMult = {value: T, blade: usize}
        //ProdTerm = {mult: T, opr1: usize, opr2: usize, ..., oprn-ary: usize}
        pub fn BinaryCayleyReturn(T: type) type {
            return struct { 
                operands_to_results: [basis_size][basis_size]BladeMult(T), 
                ///1st index = result blade, 2nd index is 0..inv_terms[result blade] independent terms that add to this result blade
                /// ProdTerm = mult{T, opr1 blade, opr2 blade, ..., oprn-ary blade}
                results_to_terms: [basis_size][basis_size]ProdTerm(T), 
                results_to_num_terms: [basis_size]usize 
            };
        }

        //step 1: create the masked cayley space
        //if this is generic to the transformation we need to tell it how to do each operand blade n-tuple
        pub fn binary_cayley_table(T: type, op: anytype, lhs_subset: usize, rhs_subset: usize) BinaryCayleyReturn(T) {
            @setEvalBranchQuota(10000000);
            //inputs to outputs under op = the image, unsimplified
            // nth axis here is the nth operand acted upon by op. the value stored here is the value returned by op given those operands
            var ret: [basis_size][basis_size]BladeMult(T) = .{.{.{.blade = 0, .value = 0.0}} ** basis_size} ** basis_size;
            //outputs to inputs under op = the pre image
            // the first axis is always the output, the second axis contains up to basis_len values containing what operand blades contributed to them
            var inv: [basis_size][basis_size]ProdTerm(T) = .{.{.{.mult = 0, .left_blade = 0, .right_blade = 0}} ** basis_size} ** basis_size;

            var inv_indices: [basis_size]usize = .{0} ** basis_size;

            for (0..basis_size) |lhs| {
                const lhs_value: T = if((lhs_subset >> lhs) & 1 != 0) 1 else 0;
                for (0..basis_size) |rhs| {

                    const rhs_value: T = if((rhs_subset >> rhs) & 1 != 0) 1 else 0;

                    const res = op.eval(
                        .{ .value = lhs_value, .blade = lhs }, 
                        .{ .value = rhs_value, .blade = rhs }
                    );
                    ret[lhs][rhs] = res;
                    if (res.value == 0) {
                        continue;
                    }
                    //@compileLog(comptimePrint("binary_cayley_table(lhs {b}, rhs {b}) = {?}", .{lhs, rhs, res}));
                    //inv[result term blade][0..n] = .{magnitude coeff, lhs operand, rhs operand}
                    // where 0..inv_indices[res.blade] are the filled in indices
                    inv[res.blade][inv_indices[res.blade]] = .{ 
                        .mult = res.value, 
                        .left_blade = lhs, 
                        .right_blade = rhs 
                    };
                    inv_indices[res.blade] += 1;
                }
            }
            //@compileLog(comptimePrint("ops_to_results {any},,,,,,,,,,,, results_to_terms {any}", .{ret, inv}));
            return .{ .operands_to_results = ret, .results_to_terms = inv, .results_to_num_terms = inv_indices };
        }


        pub fn UnaryCayleyReturn(T: type) type {
            return struct {
                operands_to_results: [basis_size]BladeMult(T),
                results_to_operands: [basis_size]UnaryProdTerm(T)
            };
        }

        pub fn unary_cayley_table(T: type, op: anytype, source_subset: usize) UnaryCayleyReturn(T) {

            var ret: [basis_size]BladeMult(T) = .{.{.blade = 0, .val = 0}} ** basis_size;
            var inv: [basis_size]UnaryProdTerm = .{.{.mult = 0, .source = 0}} ** basis_size;

            for (0..basis_size) |blade| {
                //const rhs_curr_idx = count_bits(rhs_subset & ((0b1 << rhs) - 1));
                const value: T = if((source_subset >> blade) & 1 != 0) 1 else 0;

                const res = op.eval(.{ .value = value, .blade = blade });
                ret[blade] = res;
                if (res.value == 0) {
                    continue;
                }
                //inv[result term blade][0..n] = .{magnitude coeff, lhs operand, rhs operand}
                // where 0..n are the filled in indices, n = inv_indices[]
                inv[res.blade] = .{ .mult = res.value, .opr = blade };
            }

            return .{ .operands_to_results = ret, .results_to_operands = inv};
        }


    };

    const _ret = ret;
    return _ret;
}

//impl type MVecSubset(comptime T: type, comptime alg: GAlgebraInfo, comptime runtime_subset: []const u8, comptime comptime_subset: []const u8, comptime name: ?[]const u8) type
//subset = "i,x,y,ix,iy,yx,xyz=1", name is inferred from aliases inside alg_info
// this just parses out subset and ordering info and then calls MVecSubset to actually create the type
//pub fn InferMVecSubset(alg_info: type, T: type, subset: []const u8) type {

//}

/// impl type - dense subset of the full multivector of the algebra, with runtime and comptime known parts
/// constraint: dont allow flipped basis blades relative to the algebras basis blades - thats dumb
///     if the user tries that then just create a compile error
/// DO: allow reordered terms, its trivial at the mask creation stage to reoder them as needed for an op
/// 
pub fn MVecSubset(_alg_info: type, T: type, comptime _values: []const usize) type {

    const info_args: type = _alg_info.GAlgebraInfoArgs;
    const alg_info: type = GAlgebraInfo(info_args.__dual, info_args.__basis, info_args.__ordering, info_args.__aliases);

    var _basis_values: []const BasisVector = &.{};

    var _subset: usize = 0;

    for(0.._values.len) |i| {
        const blade: usize = _values[i];

        const basis: BasisVector = alg_info.canonical_basis_with_flips[blade];

        _basis_values = _basis_values ++ .{basis};
        _subset |= (1 << blade);
    }

    const basis_size: usize = alg_info.signature.basis_size;
    //const canonical_basis_with_flips: [basis_size]BasisVector = alg_info.canonical_basis_with_flips;

    
    //TODO: find matching alias in info and make that our name if it exists
    const __name: []const u8 = "Multivector";
    const __subset = _subset;
    const __basis_values = _basis_values;
    const ret = struct {
        const Self = @This();
        pub const Algebra = alg_info;
        pub const Scalar = T;

        pub const subset = __subset;

        pub const basis_values: []const BasisVector = __basis_values;
        pub const values_arg: []const usize = _values;

        pub const num_terms = count_bits(subset);
        pub const name: []const u8 = __name;

        ///runtime int needed to fit a single blade
        pub const ud: type = std.math.IntFittingRange(0, num_terms - 1);
        ///runtime int needed to fit our maximum subset
        pub const ux: type = std.math.IntFittingRange(0, std.math.pow(u128, 2, num_terms) - 1);

        pub const BladeEnum: type = blk: {
            var fields: [num_terms]std.builtin.Type.EnumField = undefined;
            var decls = [_]std.builtin.Type.Declaration{};
            for(0..num_terms) |i| {
                const basis_vector: BasisVector = basis_values[i];
                const field = &fields[i];
                field.* = .{.name = basis_vector.name, .value = i};
            }
            break :blk @Type(.{.Enum = .{
                .tag_type = usize,
                .fields = &fields,
                .decls = &decls,
                .is_exhaustive = true,
            }});
        };
        ///takes a blade enum and returns its index in the terms array
        pub fn blade_enum_to_index(blade: BladeEnum) usize {
            for(0..num_terms) |i| {
                const basis_vector: BasisVector = basis_values[i];
                const basis_name: []const u8 = basis_vector.name;
                const blade_name: [:0]const u8 = @tagName(blade);
                var all_same: bool = true;
                for(0..basis_name.len) |j| {
                    if(blade_name[j] == 0 or (j == basis_name.len - 1 and blade_name[j + 1] != 0)) {
                        all_same = false;
                        break;
                    }
                    if(basis_name[j] != blade_name[j]) {
                        all_same = false;
                        break;
                    }
                }
                if(all_same == true) {
                    return i;
                }
            }
        }

        pub const TermsType: type = @Vector(num_terms, T);
        pub const TermsArrayType: type = [num_terms]T;
        terms: TermsType,

        //pub fn blade_array_is_correct(comptime vals: [].{BladeEnum, T}) bool {
        //    var last_index: ?usize = null;
        //    for(0..vals.len) |i| {
        //        const blade_enum: BladeEnum = vals[i].@"0";
        //        if(last_index) |last_idx| {
        //            if(last_idx > ) //this is dumb
        //        } else {
        //            last_index = blade_enum_to_index(blade_enum);
        //        }
        //        if(blade_enum_to_index(blade_enum) < i) {
        //            return false;
        //        }
        //    }
        //    return true;
        //}
        // 
        //pub fn init(comptime vals: [].{Self.BladeEnum, T}) Self {
        //    if(vals.len > )
        //}

        pub fn init_raw(values: TermsArrayType) Self {
            return Self{.terms = values};
        }

        pub fn zeros(self: *Self) *Self {
            self.terms = @splat(0);
            return self;
        }

        pub fn format(
            mvec: Self,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype
        ) !void {
            _ = try writer.print("{s} {{", .{Self.name});
            for(0..num_terms) |i| {
                _ = try writer.print("{s}: {}", .{Self.basis_values[i].name, mvec.terms[i]});
                if(i < num_terms - 1) {
                    _ = try writer.print(", ", .{});
                }
            }
            _ = try writer.print("}}", .{});
        }



        pub fn BinaryScatterMasksReturn(comptime len: usize) type {
            return struct {
                lhs_masks: [basis_size][len]i32,
                rhs_masks: [basis_size][len]i32,
                coeff_masks: [basis_size][len]T
            };
        }

        pub fn binary_create_scatter_masks_from_cayley_table_inv(comptime inv: [basis_size][basis_size]ProdTerm(T), comptime inv_indices: [basis_size]usize, comptime lhs_subset: usize, comptime rhs_subset: usize, comptime superset: usize, comptime len: usize) BinaryScatterMasksReturn(len) {
            var lhs_masks: [basis_size][len]i32 = .{.{-1} ** len} ** basis_size;
            var rhs_masks: [basis_size][len]i32 = .{.{-1} ** len} ** basis_size;
            var coeff_masks: [basis_size][len]T = .{.{0} ** len} ** basis_size;
            var masks_idx: usize = 0;
            const max_terms = basis_size;

            for(0..max_terms) |ith_term| {
                var remaining_terms: bool = false;
                for (0..basis_size) |result_blade| {

                    if ((superset & (1 << result_blade)) != 0 and inv_indices[result_blade] >= ith_term) {
                        remaining_terms = true;
                        break;
                    }
                }
                if (remaining_terms == false) {
                    break;
                }
                defer masks_idx += 1;

                var lhs_mask: [len]i32 = .{-1} ** len;
                var rhs_mask: [len]i32 = .{-1} ** len;
                var coeff_mask: [len]T = .{0} ** len;

                for (0..basis_size) |result_blade| {
                    if (superset & (1 << result_blade) == 0) {
                        continue;
                    }
                    //is this undefined?
                    const inv_term = inv[result_blade][ith_term];

                    const coeff = inv_term.mult;
                    const left_operand_blade = inv_term.left_blade;
                    const right_operand_blade = inv_term.right_blade;

                    const lhs_idx = count_bits(lhs_subset & ((1 << left_operand_blade) -% 1));
                    const rhs_idx = count_bits(rhs_subset & ((1 << right_operand_blade) -% 1));

                    const res_idx = count_bits(superset & ((1 << result_blade) -% 1));

                    lhs_mask[res_idx] = lhs_idx;
                    rhs_mask[res_idx] = rhs_idx;
                    coeff_mask[res_idx] = coeff;
                }

                @memcpy(lhs_masks[ith_term][0..len], lhs_mask[0..len]);
                @memcpy(rhs_masks[ith_term][0..len], rhs_mask[0..len]);
                @memcpy(coeff_masks[ith_term][0..len], coeff_mask[0..len]);
            }

            const ret_lhs: [basis_size]@Vector(len, i32) = lhs_masks;
            const ret_rhs: [basis_size]@Vector(len, i32) = rhs_masks;
            const ret_coeff: [basis_size]@Vector(len, T) = coeff_masks;

            return .{ .lhs_masks = ret_lhs, .rhs_masks = ret_rhs, .coeff_masks = ret_coeff };
        }

        //shuffle, mul, and add
        pub fn execute_sparse_vector_op(lhs: anytype, rhs: anytype, res: anytype, comptime len: usize, comptime masks: BinaryScatterMasksReturn(len),  comptime DataType: type) void {
            const lhs_masks = masks.lhs_masks;
            const rhs_masks = masks.rhs_masks;
            const coeff_masks = masks.coeff_masks;

            //const res_type = @typeInfo(@TypeOf(res)).Pointer.child;

            //std.debug.print("\n\t pre execute res.terms: {d:<.1}, lhs.terms: {d:<.1}, rhs.terms: {d:<.1}, ", .{res.terms, lhs.terms, rhs.terms});


            const ones: @Vector(len, T) = @splat(1.0);
            const invalid_mask: @Vector(len, i32) = @splat(-1.0);
            //const zeros: @Vector(len, T) = @splat(0.0);

            const pos: @Vector(len, T) = @splat(1);
            const neg: @Vector(len, T) = @splat(-1);

            //decreasing delta further doesnt detect near zeros.
            const delta: @Vector(len, T) = @splat(0.00001);

            inline for (lhs_masks, rhs_masks, coeff_masks) |_lhs_mask, _rhs_mask, _coeff_mask| {
                const lhs_mask: @Vector(_lhs_mask.len, i32) = _lhs_mask;
                const rhs_mask: @Vector(_lhs_mask.len, i32) = _rhs_mask;
                const coeff_mask: @Vector(_lhs_mask.len, T) = _coeff_mask;
                const invalid = comptime blk: {
                    //@compileLog(comptimePrint("lhs_mask type: {}, rhs_mask type: {}, coeff_mask type: {}", .{@TypeOf(lhs_mask), @TypeOf(rhs_mask), @TypeOf(coeff_mask)}));
                    break :blk @reduce(.And, lhs_mask == invalid_mask) or @reduce(.And, rhs_mask == invalid_mask) or @reduce(.And, @abs(coeff_mask) < delta);
                };
                if (!invalid) {
                    //@compileLog(comptimePrint("\nlhs mask: {d: <3.0}, rhs_mask: {d: <3.0}, coeff_mask: {d: <3.0}", .{ lhs_mask, rhs_mask, coeff_mask }));

                    //left: {3,7,8,9} gp right: {0,1,2,3}:
                    // for whatever reason, we determine res has {0,3,4,5,7,8,9}
                    // first = @shuffle(D, left, ones, {-1,0,-1,-1,1,2,3}) =     {1, l0, 1, 1, l1, l2, l3}
                    // second = @shuffle(D, right, ones, {0,-1,-1,1,-1,-1,-1}) = {r0, 1, 1, r1, 1, 1, 1}
                    // res.terms += {r0, l0, 1, r1, l1, l2, l3}

                    //std.debug.print("\ncoeff_mask == zeros is {}, reduce: {}", .{@as(@Vector(len, T), @splat(@as(f64, 2.0))) * coeff_mask == coeff_mask, @reduce(.And, @as(@Vector(len, T), @splat(@as(f64, 2.0))) * coeff_mask == coeff_mask)});
                    const first = @shuffle(DataType, lhs.terms, ones, lhs_mask);
                    const second = @shuffle(DataType, rhs.terms, ones, rhs_mask);
                    //std.debug.print("\n\tfirst: {d: <.1} second: {d: <.1}, coeff: {}", .{first, second, coeff_mask});
                    if (comptime @reduce(.And, coeff_mask == pos)) {
                        res.terms += first * second;
                    } else if (comptime @reduce(.And, coeff_mask == neg)) {
                        res.terms -= first * second;
                    } else {
                        res.terms += first * second * coeff_mask;
                    }
                    //std.debug.print("\n\tres.terms: {d:<.1}", .{res.terms});
                }
            }
        }


        //TODO: make this work for variably known / unknown terms
        pub fn nonzero_subset(comptime terms: [basis_size][basis_size]ProdTerm(T), comptime indices: [basis_size]usize) usize {
            _ = terms;
            var ret_subset: usize = 0;
            //const delta: T = 0.00001;
            for(0..basis_size) |blade| {
                const len = indices[blade];
                if(len == 0) {
                    continue;
                }
                //var sum: T = 0;
                //for(0..len) |i| {
                //    sum += terms[blade][i].mult;
                //}
                //if(@abs(sum) < delta) {
                //    continue;
                //}
                ret_subset |= 1 << blade;

            }
            return ret_subset;
        }

        pub fn unary_nonzero_subset(comptime terms: [basis_size]UnaryProdTerm(T)) usize {

            var ret_subset: usize = 0;
            const delta: T = 0.00001;
            for(0..basis_size) |blade| {
                const result = terms[blade];
                if(@abs(result.mult) < delta) {
                    continue;
                }
                ret_subset |= 1 << blade;

            }
            return ret_subset;
        }

        ///finds the first (if any) alias of the given algebra (_alg) that has every blade in subset in SOME order
        pub fn find_alias_matching_runtime_bitfield(comptime _alg: anytype, comptime input_subset: usize) ?GAlgebraAlias {
            const args = _alg.GAlgebraInfoArgs;
            const alg = GAlgebraInfo(args.__dual, args.__basis, args.__ordering, args.__aliases);
            for(0..alg.aliases.len) |i| {
                const alias = alg.aliases[i];
                var runtime_contains_all: bool = true;
                for(0..alias.ordering.len) |j| {
                    const alias_blade: usize = alias.ordering[j];
                    if(input_subset & (1 << alias_blade) == 0) {
                        runtime_contains_all = false;
                        break;
                    }
                } 
                if(runtime_contains_all == false) {
                    continue;
                }
                return alias;
            }

            return null;
        }

        ///given an alias, creates the corresponding _values argument usable for creating an MVecSubset matching that alias
        pub fn alias_to_values_arg(comptime alias: GAlgebraAlias) []const usize {
            var ret: []const usize = .{};
            for(0..alias.ordering.len) |i| { 
                ret = ret ++ .{alias.ordering[i]};
            }
            const _ret = ret;
            return _ret;
        }

        pub fn subset_to_values_arg(comptime sbset: usize) []const usize {

            var ret: []const usize = &.{};
            for(0..@bitSizeOf(usize)) |i| { 
                if(sbset & (1 << i) != 0) {
                    ret = ret ++ .{i};
                }   
            }
            const _ret = ret;
            return _ret;
        }

        pub fn MVecSubsetFromSubset(comptime sbset: usize) type {
            const arg = subset_to_values_arg(sbset);
            return MVecSubset(_alg_info, T, arg);
        }

        /// returns the output type of the given operation given the operand and result types.
        /// if the result type is null or void then the output type is determined by the possible blades created by the operation
        /// given the operands. if the result type is a pointer or a type then the output is forced to be that type
        pub fn BinaryRes(comptime opr: GAlgebraOp(T), comptime left_outer_type: type, comptime right_outer_type: type, comptime res_outer_type: type) type {
            const res = res_outer_type;//ResType(res_outer_type);
            //users of users of this function can pass in a pointer, type, or null for the result arg
            //pointer -> use the .child type as the logical type of the result, and fill in the results in that array
            //type -> create a new instance of that type as the result
            //null -> infer the type of the result based on opr

            //for the left and right types they can pass in a pointer or struct instance
            //pointer -> use the .child type as the logical type of that arg
            //struct -> use that structs type as the logical type, and proceed as normal

            //we should be guaranteed that res_outer_type is either .Pointer, .Struct, or .Null (@TypeOf(type) is lossy)

            const res_info = @typeInfo(res);
            //@compileLog(comptimePrint("BinaryRes: res: {s} info: {}", .{@typeName(res), res_info}));
            if(res_info == .Type) {
                @compileError(comptimePrint("unguarded type passed into BinaryRes()! @TypeOf(type) is lossy so if a user passes in a type you have to pass in that type directly as an argument instead of passing in @TypeOf(that type)", .{}));
            }
            if(res_info == .Pointer or res_info == .Struct) {
                return res;
            }

            const left_ret = UnwrapPtrType(left_outer_type);
            const right_ret = UnwrapPtrType(right_outer_type);
            
            const left_type: type = left_ret.@"0";
            const right_type: type = right_ret.@"0";
 
            const table = comptime Algebra.binary_cayley_table(T, opr.BladeOp(Algebra.signature), left_type.subset, right_type.subset);
            //comptime {
            //    var str: [1000]u8 = .{0} ** 1000;
            //    var c = 0;
            //    for(0..basis_size) |b| {
            //        const slc: []u8 = try std.fmt.bufPrint(str[c..str.len], "blade {b} =", .{b});
            //        c += slc.len;
            //        for(0..table.results_to_num_terms[b]) |n| {
            //            const fmt: []const u8 = if(n == 0) " {} from {b} * {b}" else " + {} from {b} * {b}";
            //            const term: ProdTerm(T) = table.results_to_terms[b][n];
            //
            //            const slc2: []u8 = try std.fmt.bufPrint(str[c..str.len], fmt, .{term.mult, term.left_blade, term.right_blade});
            //            c += slc2.len;
            //        }
            //        for(0..3) |_| {
            //            str[c] = ' ';
            //            c += 1;
            //        }
            //    }
            //    @compileLog(str[0..c]);
            //}
            //TODO: THIS IS WHERE THE LOGIC IS WRONG! currently it assumes that 10 * 1 = - 1 * 10 and 11 + - 11 cancels out but that only works if
            // the coefficients of lhs[1,10] = rhs[1,10]
            const ret_subset: usize = comptime nonzero_subset(table.results_to_terms, table.results_to_num_terms);
            //@compileLog(comptimePrint("BinaryRes after cayley: values: {any}, left subset: {b}, right subset: {b} ret_subset: {b}", .{_values, left_type.subset, right_type.subset, ret_subset}));
            const matching_alias = comptime find_alias_matching_runtime_bitfield(Algebra, ret_subset);
            if(matching_alias) |alias| {
                const arg = alias_to_values_arg(alias);
                return MVecSubset(_alg_info, T, arg);
            } else {
                const arg = subset_to_values_arg(ret_subset);
                //@compileLog(comptimePrint("BinaryRes subset_to_values_arg({b}) = {any}", .{ret_subset, arg}));
                return MVecSubset(_alg_info, T, arg);
            }
            //return MVecSubset(ret_subset);
        }

        pub fn UnaryRes(comptime opr: GAlgebraOp(T), source: type, drain: type) type {
            const res_info = @typeInfo(drain);

            if(res_info == .Type) {
                @compileError(comptimePrint("unguarded type passed into UnaryRes()! @TypeOf(type) is lossy so you have to pass in the type directly if a user passes in a type", .{}));
            }
            if(res_info == .Pointer or res_info == .Struct) {
                return drain;
            }

            if(res_info != .Null and res_info != .Void) {
                @compileError(comptimePrint("invalid result argument passed to UnaryRes()!"));
            }

            const source_ret = UnwrapPtrType(source);

            const source_t: type = source_ret.@"0";

            const table = comptime Algebra.unary_cayley_table(T, opr.BladeOp(Algebra.signature), source_t.subset);
            const ret_subset: usize = unary_nonzero_subset(table.results_to_operands);
            
            const matching_alias = comptime find_alias_matching_runtime_bitfield(Algebra, ret_subset);
            if(matching_alias) |alias| {
                const arg = alias_to_values_arg(alias);
                return MVecSubset(_alg_info, T, arg);
            } else {
                const arg = subset_to_values_arg(ret_subset);
                return MVecSubset(_alg_info, T, arg);
            }

            //return MVecSubset(ret_subset);

        }

        ///if passed_in_res is a pointer, returns that. otherwise creates a new instance of the true_res type passed in
        pub fn get_result(passed_in_res: anytype, comptime true_res: type) true_res {
            if(@typeInfo(@TypeOf(passed_in_res)) == .Pointer) {
                return passed_in_res;
            }
            var res: true_res = undefined;
            _=res.zeros();
            return res;
        }


        pub fn gp(left: anytype, right: anytype, res: anytype) BinaryRes(.GeometricProduct, @TypeOf(left), @TypeOf(right), if(@typeInfo(@TypeOf(res)) == .Type) res else @TypeOf(res)) {
            const Result: type = BinaryRes(.GeometricProduct, @TypeOf(left), @TypeOf(right), if(@typeInfo(@TypeOf(res)) == .Type) res else @TypeOf(res));
            var result = get_result(res, Result);

            const Left = @TypeOf(left);
            const Right = @TypeOf(right);

            const left_info = @typeInfo(Left);
            const right_info = @typeInfo(Right);   
            const res_info = @typeInfo(Result);             

            const LeftInner = if(left_info == .Pointer) left_info.Pointer.child else Left;
            const RightInner = if(right_info == .Pointer) right_info.Pointer.child else Right;
            const ResInner = if(res_info == .Pointer) res_info.Pointer.child else Result;
            

            //@compileLog(comptimePrint("gp left: {s}, right: {s},,, left ordering: {any},,, right ordering: {any},,, Result ordering {any}", .{@typeName(LeftInner), @typeName(RightInner), LeftInner.values_arg, RightInner.values_arg, ResInner.values_arg}));

            const left_set: usize = LeftInner.subset;
            const right_set: usize = RightInner.subset;
            const res_set: usize = ResInner.subset;

            ///// like BladeMult but the blade is inferred by its index in an array
            //pub fn KnownVal(T: type) type {
            //    return struct {
            //        known: bool,
            //        val: T
            //    };
            //}

            /////null val = runtime known, nonnull = comptime known
            //const RuntimeOrComptimeBlade = struct {
            //    blade: usize,
            //    val: ?f64,
            //};

            const table = comptime Algebra.binary_cayley_table(T, GAlgebraOp(T).GeometricProduct.BladeOp(Algebra.signature), left_set, right_set);
            //tell the table inverter the res size, then we can expect it to return an array with comptime known size
            const res_size = comptime count_bits(res_set);
            const operation = comptime binary_create_scatter_masks_from_cayley_table_inv(table.results_to_terms, table.results_to_num_terms, left_set, right_set, res_set, res_size);


            //@compileLog(comptimePrint("\nleft_set: {b}, right_set: {b}, res_set: {b}, table: {}, res_size: {}, operation: {}", .{left_set, right_set, res_set, table, res_size, operation}));

            const left_arg = blk: {
                if(left_info == .Pointer) {
                    break :blk left;
                }
                break :blk &left;
            };

            const res_to_use = blk: {
                if(@typeInfo(Result) == .Pointer) {
                    break :blk result;
                }
                break :blk &result;
            };

            execute_sparse_vector_op(left_arg, right, res_to_use, comptime res_size, comptime operation, T);
            return result;
        }

        pub fn Add(comptime lhs: type, comptime rhs: type, comptime res: type) type {
            const res_info = @typeInfo(res);
            if(res_info == .Type or res_info == .Pointer or res_info == .Struct) {
                return res;
            }

            const lr = comptime UnwrapPtrType(lhs);
            const rr = comptime UnwrapPtrType(rhs);

            const LeftInner: type = lr.@"0";
            const RightInner: type = rr.@"0";

            const left_set: usize = LeftInner.runtime_subset;
            const right_set: usize = RightInner.runtime_subset;

            var res_set: usize = 0;

            for(0..basis_size) |blade_i| {
                for(0..basis_size) |blade_j| {
                    if(left_set & (1 << blade_i) != 0) {
                        res_set |= (1 << blade_i);
                    }
                    if(right_set & (1 << blade_j) != 0) {
                        res_set |= (1 << blade_j);
                    }
                }
            }
            return MVecSubsetFromSubset(res_set);
        }


        pub fn add(lhs: anytype, rhs: anytype, res: anytype) Add(@TypeOf(lhs), @TypeOf(rhs), if(@typeInfo(@TypeOf(res)) == .Type) res else @TypeOf(res)) { 
            const Result = Add(@TypeOf(lhs), @TypeOf(rhs), if(@typeInfo(@TypeOf(res)) == .Type) res else @TypeOf(res));
            var result = get_result(res, Result);

            const left_type: type = UnwrapPtrType(@TypeOf(lhs)).@"0";
            const right_type: type = UnwrapPtrType(@TypeOf(rhs)).@"0";
            //@compileLog(comptimePrint("right type name: {s}, info {} ------------------------------------- rhs name {s} info {}", .{@typeName(right_type), @typeInfo(right_type), @typeName(@TypeOf(rhs)), @typeInfo(@TypeOf(rhs))}));
            const ResInner: type = UnwrapPtrType(Result).@"0";

            const left_set: usize = left_type.runtime_subset;
            const right_set: usize = comptime blk: {
                if(@typeInfo(right_type) == .Pointer) {
                    @compileError(comptimePrint("right type info {} rhs type info {}",.{@typeInfo(right_type), @typeInfo(@TypeOf(rhs))}));
                }
                break :blk right_type.runtime_subset;
            };
            const res_set: usize = comptime blk: {
                break :blk ResInner.runtime_subset;
            };
            const len = comptime count_bits(res_set);

            comptime var _lhs_mask: @Vector(len, i32) = @splat(-1);
            comptime var _rhs_mask: @Vector(len, i32) = @splat(-1);
            comptime {
                for (0..basis_size) |result_blade| {
                    if (res_set & (1 << result_blade) == 0) {
                        continue;
                    }
                    //left_set = 0b01011101, if res_blade = 101 (5)
                    //then if (0b01011101 & (0b1 << 101 = 0b100000) != 0 (true)),
                    //then the index in left for blade 101 is:
                    //count_bits(0b01011101 & (0b100000 -% 1 = 0b011111) = 0b00011101) = 4
                    //which is correct
                    
                    //may not exist
                    const left_idx = count_bits(left_set & ((1 << result_blade) -% 1));
                    const right_idx = count_bits(right_set & ((1 << result_blade) -% 1));

                    //must exist, and if an operand idx exists this is >= that
                    //assuming res is a superset of lhs.subset_field | rhs.subset_field
                    const res_idx = count_bits(res_set & ((1 << result_blade) -% 1));

                    if (left_set & (1 << result_blade) != 0) {
                        _lhs_mask[res_idx] = left_idx;
                    }
                    if (right_set & (1 << result_blade) != 0) {
                        _rhs_mask[res_idx] = right_idx;
                    }
                }
            }
            const lhs_mask = _lhs_mask;
            const rhs_mask = _rhs_mask;
            const zero: @Vector(len, T) = @splat(0);

            const left = @shuffle(T, lhs.terms, zero, lhs_mask);
            const right = @shuffle(T, rhs.terms, zero, rhs_mask);

            result.terms = left + right;

            return result;

            //inline for (0..ret_type.num_terms) |i| {
            //    const mvec_idx = comptime blk: {
            //        break :blk ret_type.mvec_index(i);
            //    };
            //    if (lhs_type.has(mvec_idx)) { //this has terrible code generation, it "inlines" by jumping all over the place
            //        if (rhs_type.has(mvec_idx)) {
            //            ret_ptr.set_redirect(mvec_idx, lhs.get_redirect(mvec_idx) + rhs.get_redirect(mvec_idx));
            //        } else {
            //            ret_ptr.set_redirect(mvec_idx, lhs.get_redirect(mvec_idx));
            //        }
            //    } else {
            //        if (rhs_type.has(mvec_idx)) {
            //            ret_ptr.set_redirect(mvec_idx, rhs.get_redirect(mvec_idx));
            //        }
            //    }
            //    //check if lhs or rhs have the given field value, and if they do do the add. if they dont then continue
            //    //we need a way of comparing
            //    //result.terms[i] = lhs.terms[i] + rhs.terms[i];
            //}
            //return ret_ptr;
        }

        pub fn sub(lhs: anytype, rhs: anytype, res: anytype) Add(@TypeOf(lhs), @TypeOf(rhs), if(@typeInfo(@TypeOf(res)) == .Type) res else @TypeOf(res)) { 
            const Result = Add(@TypeOf(lhs), @TypeOf(rhs), if(@typeInfo(@TypeOf(res)) == .Type) res else @TypeOf(res));
            var result = get_result(res, Result);

            const left_type: type = UnwrapPtrType(@TypeOf(lhs)).@"0";
            const right_type: type = UnwrapPtrType(@TypeOf(rhs)).@"0";
            const ResInner: type = UnwrapPtrType(Result).@"0";

            const left_set: usize = comptime blk: {
                break :blk left_type.runtime_subset;
            };
            const right_set: usize = comptime blk: {
                break :blk right_type.runtime_subset;
            };
            const res_set: usize = comptime blk: {
                break :blk ResInner.runtime_subset;
            };
            const len = comptime count_bits(res_set);

            comptime var _lhs_mask: @Vector(len, i32) = @splat(-1);
            comptime var _rhs_mask: @Vector(len, i32) = @splat(-1);
            comptime {
                for (0..basis_size) |result_blade| {
                    if (res_set & (1 << result_blade) == 0) {
                        continue;
                    }
                    //left_set = 0b01011101, if res_blade = 101 (5)
                    //then if (0b01011101 & (0b1 << 101 = 0b100000) != 0 (true)),
                    //then the index in left for blade 101 is:
                    //count_bits(0b01011101 & (0b100000 -% 1 = 0b011111) = 0b00011101) = 4
                    //which is correct
                    
                    //may not exist
                    const left_idx = count_bits(left_set & ((1 << result_blade) -% 1));
                    const right_idx = count_bits(right_set & ((1 << result_blade) -% 1));

                    //must exist, and if an operand idx exists this is >= that
                    //assuming res is a superset of lhs.subset_field | rhs.subset_field
                    const res_idx = count_bits(res_set & ((1 << result_blade) -% 1));

                    if (left_set & (1 << result_blade) != 0) {
                        _lhs_mask[res_idx] = left_idx;
                    }
                    if (right_set & (1 << result_blade) != 0) {
                        _rhs_mask[res_idx] = right_idx;
                    }
                }
            }
            const lhs_mask = _lhs_mask;
            const rhs_mask = _rhs_mask;
            const zero: @Vector(len, T) = @splat(0);

            const left = @shuffle(T, lhs.terms, zero, lhs_mask);
            const right = @shuffle(T, rhs.terms, zero, rhs_mask);

            result.terms = left - right;

            return result;
        }

        pub fn mul_scalar(self: anytype, scalar: T, res: anytype) BinaryRes(.GeometricProduct, @TypeOf(self), MVecSubset(1), if(@typeInfo(@TypeOf(res)) == .Type) res else @TypeOf(res)) {
            //@compileLog(comptimePrint("mul_scalar self: {s}, res name: {s}, res type info: {?}", .{@typeName(@TypeOf(self)), @typeName(@TypeOf(res)), @typeInfo(@TypeOf(res))}));
            const Result = BinaryRes(.GeometricProduct, @TypeOf(self), MVecSubset(1), if(@typeInfo(@TypeOf(res)) == .Type) res else @TypeOf(res));
            //var result = get_result(res, Result);
            var result = blk: {

                const passed_res_info = blk2: {
                    const ti = @typeInfo(@TypeOf(res));
                    if(ti == .Type) {
                        break :blk2 @typeInfo(res);
                    }
                    break :blk2 ti;
                };
                if(passed_res_info == .Pointer) {
                    //const ret: true_res = @as(passed_in_res, true_res);
                    //@compileLog(comptimePrint("passed_res_info child: {?}, true res: {s}", .{passed_res_info.Pointer.child, @typeName(true_res)}));
                    break :blk res;
                }
                var ret: Result = undefined;
                _=ret.zeros();
                break :blk ret;
            };
            const result_ref = if(@typeInfo(@TypeOf(result)) == .Pointer) result else &result;
            _mul_scalar(self, scalar, result_ref);
            return result;
        }

        pub fn _mul_scalar(self: anytype, scalar: T, result: anytype) void {
            const Result: type = @TypeOf(result);

            if(Result == Self or Result == *Self) {
                result.terms = self.terms * @as(@TypeOf(self.terms), @splat(scalar));
            } else {
                const ResInner: type = UnwrapPtrType(Result).@"0";
                inline for(0..basis_size) |blade_t| {
                    //const blade_t: ud = @truncate(blade);
                    if(ResInner.subset_field & (1 << blade_t) != 0 and Self.subset_field & (1 << blade_t) != 0) {
                        const us_idx = comptime count_bits(Self.subset_field & ((1 << blade_t) -% 1));
                        const res_idx = comptime count_bits(ResInner.subset_field & ((1 << blade_t) -% 1));
                        result.terms[res_idx] = self.terms[us_idx] * scalar;
                    }
                }
            }
        }

    };
    return ret;
}

//requirements: 
//  need to be able to separate logic from implementation
//  need to be able to specify a dual or non dual algebra
//  implementation types need to know about each other when theyre part of the same algebra (implies a wrapping impl type around MVecSubset)
//  need to be able to name basis vectors whatever i want (x,y,z,e012), but also what those basis vectors square to (0i,+x,+y,+z,-t)
//  need to be able to define a standard ordering
//  need to be able to name specific subsets in the given algebra, each with their own ordering and possibly additional basis blade alias's ("Motor", "s,ix,iy,iz,xy,yz,zx")
//  implementors and consumers of those implementors need to be able to easily iterate through mvec arrays / find what basis blades correspond to an index
//  logic type must have a function equivalent to cayley_table in the original impl, but 1. comptime known vals work, and 2. different orderings of the basis blades work
//  the outer impl type should not be the horizon of what mvecs can operate with, the algebra that was defined could always be a subset of some other algebra with a larger signature

//idea 1:
//NO outer logic type that just takes (dual, p,z,n), but a bunch of functions that impls can use at comptime to create the data transformations they want at runtime
//const BasisBlade = struct {val: f64 = 1.0, metric: BasisMetric, binblade: usize} //should be generic to value?
//
//const GAOperation = enum{GeometricProduct, OuterProduct,...}
//
//(and another for unary ops)
//[for impls] determine the logical type both operands belong to -> create comptime basisblade arrays representing the values of each operand known at compile time not to be 0
//-> pass those to nary_cayley_table() -> get a tuple {result blade @ index -> 0..k [n operand blades] terms}
//-> given that, the impl type (logic types dont know about arrays) creates the scatter masks for lhs and rhs for the <=k terms in the result
//-> impl type calls a componentwise scatter add function given those masks
//if lhs term[i] is e21, and rhs term[j] is e12, but result[res idx] is in e12, then you do result[idx] += op(e21,e12)
//pub fn binary_cayley_table(comptime lhs: []BasisBlade, comptime rhs: []BasisBlade, comptime res_basis: []BasisBlade, comptime operation: Operation)

//const GAAlias = struct {
//  name: ?[]const u8, //= "Multivector", ?
//  runtime_subset: []const BasisVector,
//  comptime_subset: []const BasisVector,
//  specific_aliases: ?[]const struct{BasisVector, []const u8},
//
//
//}
//const pga3_aliases = .{.{name: "Motor", runtime_subset: "s,ix,iy,iz,xy,yz,zx"}, 
//                       .{name: "Point", runtime_subset: "izy,ixz,iyx,xyz", specific_aliases: "izy=x,ixz=y,iyx=z,xyz=i"},
//                       .{name: "NormPoint", runtime_subset: "izy,ixz,iyx", comp_subset: "xyz=1", specific_aliases: "izy=x,ixz=y,iyx=z,xyz=i"},
//                       .{name: "Line", runtime_subset: "ix,iy,iz,xy,yz,zx"},
//                       .{name: "IdealLine", runtime_subset: "ix,iy,iz"}}
// space info type
//fn GAlgebraInfo(dual = true, basis = "0i,+x,+y,+z", ordering="",aliases=pga3_aliases, use_e = false)
//
// actual runtime type. i think i just have to accept that the signature is going to be long
//impl type MVecSubset(comptime T: type, comptime alg: GAlgebraInfo, comptime runtime_subset: []const u8, comptime comptime_subset: []const u8, comptime name: ?[]const u8) type
//      //used to interface with logic fn's
//      const basis_blades: [basis_len]BasisBlade = create_basis_array(...)
//      terms: @Vector(T, our_len)
//
//      pub fn create_scatter_masks(...)
//      pub fn componentwise_multiscatter_add(...) void
//
//      pub fn BinaryResult(lhs: type, rhs: type, res: type, opr: GAOperation) type
//          when inferring res type, do what the old impl did except when inferring the output 
//          and its signature matches an alias'd type, use that alias specifically
//      pub fn get_result(...)
//
//      impl every op
//      pub fn gp(lhs: anytype, rhs: anytype, res: anytype) BinaryResult(...)
//          get basis blades of all
//          var res = get_result(...)
//          const inv = comptime ga_logic.binary_cayley_table(lhs_basis,rhs_basis,res_basis, GAOperation.GeometricProduct)
//          const opr = comptime create_scatter_masks(inv)
//          componentwise_multiscatter_add(&lhs,&rhs,&res,opr);
//          return res
   

//the algebra doesnt care at all about any basis and it just operates on whatever basis ordering you give it
//the implementors need to reorder ops info themselves
/// Geometric Algebra logical type, contains only information needed to define a geometric algebra with the given signature
/// that is unique up to isomorphism, as well as helper procs related to that.
/// it knows nothing about the properties or names you assert about the basis vectors nor what ordering theyre in
pub fn GAlgebra(comptime _p: usize, comptime _n: usize, comptime _z: usize) type {
    const _d: usize = _p + _n + _z;
    const _basis_len: usize = std.math.pow(usize, 2, _d);

    var __z = _z;
    var __p = _p;
    var __n = _n;

    var _zero_mask: usize = 0;
    var _pos_mask: usize = 0;
    var _neg_mask: usize = 0;
    var curr_metric: ?isize = 0;
    for(0.._d) |axis| {
        if(curr_metric == 0) {
            if(__z > 0) {
                __z -= 1;
                _zero_mask |= (0b1 << axis);
                continue;
            } else {
                curr_metric = 1;
            }
        }
        if(curr_metric > 0) {
            if(__p > 0) {
                __p -= 1;
                _pos_mask |= (0b1 << axis);
                continue;
            } else {
                curr_metric = -1;
            }
        }
        if(curr_metric < 0) {
            if(__n > 0) {
                __n -= 1;
                _neg_mask |= (0b1 << axis);
                continue;
            } else {
                curr_metric = null;
                continue;
            }
        }
        if(curr_metric == null) {
            @compileError("idk how we're still creating the mask in GAlgebra");
        }
    }

    const __zm = _zero_mask;
    const __nm = _neg_mask;
    const __pm = _pos_mask;

    const alg: type = struct {
        const Self = @This();
        ///zero squaring basis vectors
        pub const z = _z;
        ///positive squaring basis vectors
        pub const p = _p;
        ///negative squaring basis vectors
        pub const n = _n;
        ///dimension
        pub const d = _d;
        //pub const options = _options;
        //pub const dual: bool = options.dual;
        pub const basis_len: usize = _basis_len;

        pub const zero_mask: usize = __zm;
        pub const pos_mask: usize = __pm;
        pub const neg_mask: usize = __nm;

        //runtime int needed to fit a single blade (i dont think i need these)
        //pub const ud: type = std.math.IntFittingRange(0, basis_len - 1);
        //runtime int needed to fit our maximum subset
        //pub const ux: type = std.math.IntFittingRange(0, std.math.pow(u128, 2, basis_len) - 1);

        //TODO: function that returns ops of the union algebra of ourselves and some other algebra
        //pub fn GetOps()

        //pub const ClosedOps: type = GAlgebraOps(pos_mask, zero_mask, neg_mask);


    };
    return alg;
}

// implementation type of a dense subset of a multivector following some GAlgebra
// point: the logical data transformations are dependent only on the metric (incl. dimensionality) + whether the algebra is dual or not
// consideration: what if i want to permute the orders of basis elements?
// consideration: what if i want to name the basis elements?
// consideration: what if i want to select blades based on defined properties? e.g. euclidian blades
// if all of the above are parametric, how do i find the geometric product of any 2 multivectors with different values of all of these, including subset?
//pub fn MVec(comptime alg: type, comptime T: type, )

pub fn main() !void {
    //const sig: [2]BasisVector = {}
    //const galg = GAlgebraFull(2, .{.{.name = &.{'x'}, .square = .positive}, .{.name = &.{'y'}, .square = .positive}});
    //galg.test_func();
    //debug.print("AAAAAAAA", .{});
    const ordering: []const u8 = "ixy,xy,s,xi,iy,y,x,i";
    const galg = GAlgebraInfo(true, "0i,+x,+y", ordering, &.{});
    const MVecT = MVecSubset(galg, f64, &.{0, 1, 2, 4});
    @compileLog(comptimePrint("MVecT subset: {b}", .{MVecT.subset}));
    const mv1: MVecT = comptime MVecT.init_raw(.{1.0, 2.0, 3.0, 4.0});
    const mv2: MVecT = comptime MVecT.init_raw(.{3.2, -4.1, 1.2, 7.1});
    @compileLog(comptimePrint("mv1: {}, mv2: {}", .{mv1, mv2}));
    const mv3 = comptime mv1.gp(mv2, null);
    //debug.print("{} * {} = {}\n", .{mv1, mv2, mv3});
    @compileLog(comptimePrint("{} * {} = {}", .{mv1, mv2, mv3}));
}