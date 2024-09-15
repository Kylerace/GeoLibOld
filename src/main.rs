use std::{fmt, ops::{Add, BitAnd, BitOr, BitXor, Index, IndexMut, Mul, Not, Sub}};


type Scalar = f64;

const basis: &'static [&'static str] = &[ "1","e0","e1","e2","e3","e01","e02","e03","e12","e31","e23","e021","e013","e032","e123","e0123" ];
const basis_count: usize = basis.len();

// basis vectors are available as global constants.
const e0: MVec = MVec{terms: [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]};
const e1: MVec = MVec{terms: [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]};
const e2: MVec = MVec{terms: [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]};
const e3: MVec = MVec{terms: [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]};
const e01: MVec = MVec{terms: [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]};
const e02: MVec = MVec{terms: [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]};
const e03: MVec = MVec{terms: [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.]};
const e12: MVec = MVec{terms: [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.]};
const e31: MVec = MVec{terms: [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]};
const e23: MVec = MVec{terms: [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.]};
const e021: MVec = MVec{terms: [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.]};
const e013: MVec = MVec{terms: [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]};
const e032: MVec = MVec{terms: [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.]};
const e123: MVec = MVec{terms: [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]};
const e0123: MVec = MVec{terms: [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]};

const si: usize = 0;
const e0i: usize = 1;
const e1i: usize = 2;
const e2i: usize = 3;
const e3i: usize = 4;
const e01i: usize = 5;
const e02i: usize = 6;
const e03i: usize = 7;
const e12i: usize = 8;
const e31i: usize = 9;
const e23i: usize = 10;
const e021i: usize = 11;
const e013i: usize = 12;
const e032i: usize = 13;
const e123i: usize = 14;
const e0123i: usize = 15;

enum Basis {
    s,
    e0,
    e1,
    e2,
    e3,
    e01,
    e02,
    e30,
    e12,
    e31,
    e23,
    e021,
    e013,
    e032,
    e123,
    e0123
}

const PI: Scalar = std::f64::consts::PI;//3.14159265358979;

pub trait VectorSpaceElement {
    fn add(&self, rhs: &Self) -> Self;
    fn scalar_multiply(&self, rhs: &Self) -> Self;
}

pub trait InnerProductSpaceElement: VectorSpaceElement {
    type Output;
    fn inner_product(&self, rhs: &Self) -> Self::Output;
    fn norm(&self) -> Scalar;
    fn normalize(&self) -> Self;
}

pub trait GeometricAlgebraElement: Index<usize> + IndexMut<usize> + InnerProductSpaceElement + Sized {
    type MultiVector: GeometricAlgebraElement;

    fn outer_product(&self, b: &Self) -> Self;
    fn geometric_product(&self, b: &Self) -> Self;

    fn join(&self, b: &Self) -> Self;
    fn meet(&self, b: &Self) -> Self;

    fn inverse(&self) -> Option<Self>;
    fn reverse(&self) -> Self;
    fn dual(&self) -> Self;
    fn conjugate(&self) -> Self;
    fn involute(&self) -> Self;
    
}


#[derive(Default,Debug,Clone,Copy,PartialEq)]
struct MVec {
    pub terms: [Scalar; basis_count]
}

impl MVec {
    pub fn zero() -> Self {
        MVec{terms: [0.0; basis_count]}
    }

    pub fn new(v: Scalar, i: usize) -> Self {
        let mut ret = Self::zero();
        ret[i] = v;
        ret
    }
}

impl Index<usize> for MVec {
    type Output = Scalar;

    fn index<'a>(&'a self, index: usize) -> &'a Self::Output {
        &self.terms[index]
    }
}

impl IndexMut<usize> for MVec {
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut Self::Output {
        &mut self.terms[index]
    }
}

impl fmt::Display for MVec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut n = 0;
        let ret = self.terms.iter().enumerate().filter_map(|(i, &coeff)| {
            if coeff > 0.00001 || coeff < -0.00001 {
                n = 1;
                Some(format!("{}{}", 
                        format!("{:.*}", 7, coeff).trim_end_matches('0').trim_end_matches('.'),
                        if i > 0 { basis[i] } else { "" }
                    )
                )
            } else {
                None
            }
        }).collect::<Vec<String>>().join(" + ");
        if n==0 { write!(f,"0") } else { write!(f, "{}", ret) }
    }
}

// Reverse
// Reverse the order of the basis blades.
impl MVec {
    pub fn Reverse(&self) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0]=a[0];
        res[1]=a[1];
        res[2]=a[2];
        res[3]=a[3];
        res[4]=a[4];
        res[5]= -a[5];
        res[6]= -a[6];
        res[7]= -a[7];
        res[8]= -a[8];
        res[9]= -a[9];
        res[10]= -a[10];
        res[11]= -a[11];
        res[12]= -a[12];
        res[13]= -a[13];
        res[14]= -a[14];
        res[15]=a[15];
        res
    }
}

// Dual
// Poincare duality operator.
impl MVec {
    pub fn dual(&self) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0]=a[15];
        res[1]=a[14];
        res[2]=a[13];
        res[3]=a[12];
        res[4]=a[11];
        res[5]=a[10];
        res[6]=a[9];
        res[7]=a[8];
        res[8]=a[7];
        res[9]=a[6];
        res[10]=a[5];
        res[11]=a[4];
        res[12]=a[3];
        res[13]=a[2];
        res[14]=a[1];
        res[15]=a[0];
        res
    }
}

impl Not for MVec {
    type Output = MVec;

    fn not(self) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0]=a[15];
        res[1]=a[14];
        res[2]=a[13];
        res[3]=a[12];
        res[4]=a[11];
        res[5]=a[10];
        res[6]=a[9];
        res[7]=a[8];
        res[8]=a[7];
        res[9]=a[6];
        res[10]=a[5];
        res[11]=a[4];
        res[12]=a[3];
        res[13]=a[2];
        res[14]=a[1];
        res[15]=a[0];
        res
    }
}

// Conjugate
// Clifford Conjugation
impl MVec {
    pub fn Conjugate(&self) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0]=a[0];
        res[1]= -a[1];
        res[2]= -a[2];
        res[3]= -a[3];
        res[4]= -a[4];
        res[5]= -a[5];
        res[6]= -a[6];
        res[7]= -a[7];
        res[8]= -a[8];
        res[9]= -a[9];
        res[10]= -a[10];
        res[11]=a[11];
        res[12]=a[12];
        res[13]=a[13];
        res[14]=a[14];
        res[15]=a[15];
        res
    }
}

// Involute
// Main involution
impl MVec {
    pub fn Involute(&self) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0]=a[0];
        res[1]= -a[1];
        res[2]= -a[2];
        res[3]= -a[3];
        res[4]= -a[4];
        res[5]=a[5];
        res[6]=a[6];
        res[7]=a[7];
        res[8]=a[8];
        res[9]=a[9];
        res[10]=a[10];
        res[11]= -a[11];
        res[12]= -a[12];
        res[13]= -a[13];
        res[14]= -a[14];
        res[15]=a[15];
        res
    }
}

// Mul
// The geometric product.
impl Mul for MVec {
    type Output = MVec;

    fn mul(self: MVec, b: MVec) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0]=b[0]*a[0]+b[2]*a[2]+b[3]*a[3]+b[4]*a[4]-b[8]*a[8]-b[9]*a[9]-b[10]*a[10]-b[14]*a[14];
		res[1]=b[1]*a[0]+b[0]*a[1]-b[5]*a[2]-b[6]*a[3]-b[7]*a[4]+b[2]*a[5]+b[3]*a[6]+b[4]*a[7]+b[11]*a[8]+b[12]*a[9]+b[13]*a[10]+b[8]*a[11]+b[9]*a[12]+b[10]*a[13]+b[15]*a[14]-b[14]*a[15];
		res[2]=b[2]*a[0]+b[0]*a[2]-b[8]*a[3]+b[9]*a[4]+b[3]*a[8]-b[4]*a[9]-b[14]*a[10]-b[10]*a[14];
		res[3]=b[3]*a[0]+b[8]*a[2]+b[0]*a[3]-b[10]*a[4]-b[2]*a[8]-b[14]*a[9]+b[4]*a[10]-b[9]*a[14];
		res[4]=b[4]*a[0]-b[9]*a[2]+b[10]*a[3]+b[0]*a[4]-b[14]*a[8]+b[2]*a[9]-b[3]*a[10]-b[8]*a[14];
		res[5]=b[5]*a[0]+b[2]*a[1]-b[1]*a[2]-b[11]*a[3]+b[12]*a[4]+b[0]*a[5]-b[8]*a[6]+b[9]*a[7]+b[6]*a[8]-b[7]*a[9]-b[15]*a[10]-b[3]*a[11]+b[4]*a[12]+b[14]*a[13]-b[13]*a[14]-b[10]*a[15];
		res[6]=b[6]*a[0]+b[3]*a[1]+b[11]*a[2]-b[1]*a[3]-b[13]*a[4]+b[8]*a[5]+b[0]*a[6]-b[10]*a[7]-b[5]*a[8]-b[15]*a[9]+b[7]*a[10]+b[2]*a[11]+b[14]*a[12]-b[4]*a[13]-b[12]*a[14]-b[9]*a[15];
		res[7]=b[7]*a[0]+b[4]*a[1]-b[12]*a[2]+b[13]*a[3]-b[1]*a[4]-b[9]*a[5]+b[10]*a[6]+b[0]*a[7]-b[15]*a[8]+b[5]*a[9]-b[6]*a[10]+b[14]*a[11]-b[2]*a[12]+b[3]*a[13]-b[11]*a[14]-b[8]*a[15];
		res[8]=b[8]*a[0]+b[3]*a[2]-b[2]*a[3]+b[14]*a[4]+b[0]*a[8]+b[10]*a[9]-b[9]*a[10]+b[4]*a[14];
		res[9]=b[9]*a[0]-b[4]*a[2]+b[14]*a[3]+b[2]*a[4]-b[10]*a[8]+b[0]*a[9]+b[8]*a[10]+b[3]*a[14];
		res[10]=b[10]*a[0]+b[14]*a[2]+b[4]*a[3]-b[3]*a[4]+b[9]*a[8]-b[8]*a[9]+b[0]*a[10]+b[2]*a[14];
		res[11]=b[11]*a[0]-b[8]*a[1]+b[6]*a[2]-b[5]*a[3]+b[15]*a[4]-b[3]*a[5]+b[2]*a[6]-b[14]*a[7]-b[1]*a[8]+b[13]*a[9]-b[12]*a[10]+b[0]*a[11]+b[10]*a[12]-b[9]*a[13]+b[7]*a[14]-b[4]*a[15];
		res[12]=b[12]*a[0]-b[9]*a[1]-b[7]*a[2]+b[15]*a[3]+b[5]*a[4]+b[4]*a[5]-b[14]*a[6]-b[2]*a[7]-b[13]*a[8]-b[1]*a[9]+b[11]*a[10]-b[10]*a[11]+b[0]*a[12]+b[8]*a[13]+b[6]*a[14]-b[3]*a[15];
		res[13]=b[13]*a[0]-b[10]*a[1]+b[15]*a[2]+b[7]*a[3]-b[6]*a[4]-b[14]*a[5]-b[4]*a[6]+b[3]*a[7]+b[12]*a[8]-b[11]*a[9]-b[1]*a[10]+b[9]*a[11]-b[8]*a[12]+b[0]*a[13]+b[5]*a[14]-b[2]*a[15];
		res[14]=b[14]*a[0]+b[10]*a[2]+b[9]*a[3]+b[8]*a[4]+b[4]*a[8]+b[3]*a[9]+b[2]*a[10]+b[0]*a[14];
		res[15]=b[15]*a[0]+b[14]*a[1]+b[13]*a[2]+b[12]*a[3]+b[11]*a[4]+b[10]*a[5]+b[9]*a[6]+b[8]*a[7]+b[7]*a[8]+b[6]*a[9]+b[5]*a[10]-b[4]*a[11]-b[3]*a[12]-b[2]*a[13]-b[1]*a[14]+b[0]*a[15];
        res
    }
}

// Wedge
// The outer product. (MEET)
impl BitXor for MVec {
    type Output = MVec;

    fn bitxor(self: MVec, b: MVec) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0]=b[0]*a[0];
		res[1]=b[1]*a[0]+b[0]*a[1];
		res[2]=b[2]*a[0]+b[0]*a[2];
		res[3]=b[3]*a[0]+b[0]*a[3];
		res[4]=b[4]*a[0]+b[0]*a[4];
		res[5]=b[5]*a[0]+b[2]*a[1]-b[1]*a[2]+b[0]*a[5];
		res[6]=b[6]*a[0]+b[3]*a[1]-b[1]*a[3]+b[0]*a[6];
		res[7]=b[7]*a[0]+b[4]*a[1]-b[1]*a[4]+b[0]*a[7];
		res[8]=b[8]*a[0]+b[3]*a[2]-b[2]*a[3]+b[0]*a[8];
		res[9]=b[9]*a[0]-b[4]*a[2]+b[2]*a[4]+b[0]*a[9];
		res[10]=b[10]*a[0]+b[4]*a[3]-b[3]*a[4]+b[0]*a[10];
		res[11]=b[11]*a[0]-b[8]*a[1]+b[6]*a[2]-b[5]*a[3]-b[3]*a[5]+b[2]*a[6]-b[1]*a[8]+b[0]*a[11];
		res[12]=b[12]*a[0]-b[9]*a[1]-b[7]*a[2]+b[5]*a[4]+b[4]*a[5]-b[2]*a[7]-b[1]*a[9]+b[0]*a[12];
		res[13]=b[13]*a[0]-b[10]*a[1]+b[7]*a[3]-b[6]*a[4]-b[4]*a[6]+b[3]*a[7]-b[1]*a[10]+b[0]*a[13];
		res[14]=b[14]*a[0]+b[10]*a[2]+b[9]*a[3]+b[8]*a[4]+b[4]*a[8]+b[3]*a[9]+b[2]*a[10]+b[0]*a[14];
		res[15]=b[15]*a[0]+b[14]*a[1]+b[13]*a[2]+b[12]*a[3]+b[11]*a[4]+b[10]*a[5]+b[9]*a[6]+b[8]*a[7]+b[7]*a[8]+b[6]*a[9]+b[5]*a[10]-b[4]*a[11]-b[3]*a[12]-b[2]*a[13]-b[1]*a[14]+b[0]*a[15];
        res
    }
}

// Vee
// The regressive product. (JOIN)
impl BitAnd for MVec {
    type Output = MVec;

    fn bitand(self: MVec, b: MVec) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[15]=1.0*(a[15]*b[15]);
		res[14]=-1.0*(a[14]*-1.*b[15]+a[15]*b[14]*-1.);
		res[13]=-1.0*(a[13]*-1.*b[15]+a[15]*b[13]*-1.);
		res[12]=-1.*(a[12]*-1.*b[15]+a[15]*b[12]*-1.);
		res[11]=-1.*(a[11]*-1.*b[15]+a[15]*b[11]*-1.);
		res[10]=1.*(a[10]*b[15]+a[13]*-1.*b[14]*-1.-a[14]*-1.*b[13]*-1.+a[15]*b[10]);
		res[9]=1.*(a[9]*b[15]+a[12]*-1.*b[14]*-1.-a[14]*-1.*b[12]*-1.+a[15]*b[9]);
		res[8]=1.*(a[8]*b[15]+a[11]*-1.*b[14]*-1.-a[14]*-1.*b[11]*-1.+a[15]*b[8]);
		res[7]=1.*(a[7]*b[15]+a[12]*-1.*b[13]*-1.-a[13]*-1.*b[12]*-1.+a[15]*b[7]);
		res[6]=1.*(a[6]*b[15]-a[11]*-1.*b[13]*-1.+a[13]*-1.*b[11]*-1.+a[15]*b[6]);
		res[5]=1.*(a[5]*b[15]+a[11]*-1.*b[12]*-1.-a[12]*-1.*b[11]*-1.+a[15]*b[5]);
		res[4]=1.*(a[4]*b[15]-a[7]*b[14]*-1.+a[9]*b[13]*-1.-a[10]*b[12]*-1.-a[12]*-1.*b[10]+a[13]*-1.*b[9]-a[14]*-1.*b[7]+a[15]*b[4]);
		res[3]=1.*(a[3]*b[15]-a[6]*b[14]*-1.-a[8]*b[13]*-1.+a[10]*b[11]*-1.+a[11]*-1.*b[10]-a[13]*-1.*b[8]-a[14]*-1.*b[6]+a[15]*b[3]);
		res[2]=1.*(a[2]*b[15]-a[5]*b[14]*-1.+a[8]*b[12]*-1.-a[9]*b[11]*-1.-a[11]*-1.*b[9]+a[12]*-1.*b[8]-a[14]*-1.*b[5]+a[15]*b[2]);
		res[1]=1.*(a[1]*b[15]+a[5]*b[13]*-1.+a[6]*b[12]*-1.+a[7]*b[11]*-1.+a[11]*-1.*b[7]+a[12]*-1.*b[6]+a[13]*-1.*b[5]+a[15]*b[1]);
		res[0]=1.*(a[0]*b[15]+a[1]*b[14]*-1.+a[2]*b[13]*-1.+a[3]*b[12]*-1.+a[4]*b[11]*-1.+a[5]*b[10]+a[6]*b[9]+a[7]*b[8]+a[8]*b[7]+a[9]*b[6]+a[10]*b[5]-a[11]*-1.*b[4]-a[12]*-1.*b[3]-a[13]*-1.*b[2]-a[14]*-1.*b[1]+a[15]*b[0]);
        res
    }
}

// Dot
// The inner product.
impl BitOr for MVec {
    type Output = MVec;

    fn bitor(self: MVec, b: MVec) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0]=b[0]*a[0]+b[2]*a[2]+b[3]*a[3]+b[4]*a[4]-b[8]*a[8]-b[9]*a[9]-b[10]*a[10]-b[14]*a[14];
		res[1]=b[1]*a[0]+b[0]*a[1]-b[5]*a[2]-b[6]*a[3]-b[7]*a[4]+b[2]*a[5]+b[3]*a[6]+b[4]*a[7]+b[11]*a[8]+b[12]*a[9]+b[13]*a[10]+b[8]*a[11]+b[9]*a[12]+b[10]*a[13]+b[15]*a[14]-b[14]*a[15];
		res[2]=b[2]*a[0]+b[0]*a[2]-b[8]*a[3]+b[9]*a[4]+b[3]*a[8]-b[4]*a[9]-b[14]*a[10]-b[10]*a[14];
		res[3]=b[3]*a[0]+b[8]*a[2]+b[0]*a[3]-b[10]*a[4]-b[2]*a[8]-b[14]*a[9]+b[4]*a[10]-b[9]*a[14];
		res[4]=b[4]*a[0]-b[9]*a[2]+b[10]*a[3]+b[0]*a[4]-b[14]*a[8]+b[2]*a[9]-b[3]*a[10]-b[8]*a[14];
		res[5]=b[5]*a[0]-b[11]*a[3]+b[12]*a[4]+b[0]*a[5]-b[15]*a[10]-b[3]*a[11]+b[4]*a[12]-b[10]*a[15];
		res[6]=b[6]*a[0]+b[11]*a[2]-b[13]*a[4]+b[0]*a[6]-b[15]*a[9]+b[2]*a[11]-b[4]*a[13]-b[9]*a[15];
		res[7]=b[7]*a[0]-b[12]*a[2]+b[13]*a[3]+b[0]*a[7]-b[15]*a[8]-b[2]*a[12]+b[3]*a[13]-b[8]*a[15];
		res[8]=b[8]*a[0]+b[14]*a[4]+b[0]*a[8]+b[4]*a[14];
		res[9]=b[9]*a[0]+b[14]*a[3]+b[0]*a[9]+b[3]*a[14];
		res[10]=b[10]*a[0]+b[14]*a[2]+b[0]*a[10]+b[2]*a[14];
		res[11]=b[11]*a[0]+b[15]*a[4]+b[0]*a[11]-b[4]*a[15];
		res[12]=b[12]*a[0]+b[15]*a[3]+b[0]*a[12]-b[3]*a[15];
		res[13]=b[13]*a[0]+b[15]*a[2]+b[0]*a[13]-b[2]*a[15];
		res[14]=b[14]*a[0]+b[0]*a[14];
		res[15]=b[15]*a[0]+b[0]*a[15];
        res
    }
}

// Add
// Multivector addition
impl Add for MVec {
    type Output = MVec;

    fn add(self: MVec, b: MVec) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0] = a[0]+b[0];
		res[1] = a[1]+b[1];
		res[2] = a[2]+b[2];
		res[3] = a[3]+b[3];
		res[4] = a[4]+b[4];
		res[5] = a[5]+b[5];
		res[6] = a[6]+b[6];
		res[7] = a[7]+b[7];
		res[8] = a[8]+b[8];
		res[9] = a[9]+b[9];
		res[10] = a[10]+b[10];
		res[11] = a[11]+b[11];
		res[12] = a[12]+b[12];
		res[13] = a[13]+b[13];
		res[14] = a[14]+b[14];
		res[15] = a[15]+b[15];
        res
    }
}

// Sub
// Multivector subtraction
impl Sub for MVec {
    type Output = MVec;

    fn sub(self: MVec, b: MVec) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0] = a[0]-b[0];
		res[1] = a[1]-b[1];
		res[2] = a[2]-b[2];
		res[3] = a[3]-b[3];
		res[4] = a[4]-b[4];
		res[5] = a[5]-b[5];
		res[6] = a[6]-b[6];
		res[7] = a[7]-b[7];
		res[8] = a[8]-b[8];
		res[9] = a[9]-b[9];
		res[10] = a[10]-b[10];
		res[11] = a[11]-b[11];
		res[12] = a[12]-b[12];
		res[13] = a[13]-b[13];
		res[14] = a[14]-b[14];
		res[15] = a[15]-b[15];
        res
    }
}

// smul
// scalar/multivector multiplication
impl Mul<MVec> for Scalar {
    type Output = MVec;

    fn mul(self: Scalar, b: MVec) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0] = a*b[0];
        res[1] = a*b[1];
        res[2] = a*b[2];
        res[3] = a*b[3];
        res[4] = a*b[4];
        res[5] = a*b[5];
        res[6] = a*b[6];
        res[7] = a*b[7];
        res[8] = a*b[8];
        res[9] = a*b[9];
        res[10] = a*b[10];
        res[11] = a*b[11];
        res[12] = a*b[12];
        res[13] = a*b[13];
        res[14] = a*b[14];
        res[15] = a*b[15];
        res
    }
}

// muls
// multivector/scalar multiplication
impl Mul<Scalar> for MVec {
    type Output = MVec;

    fn mul(self: MVec, b: Scalar) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0] = a[0]*b;
        res[1] = a[1]*b;
        res[2] = a[2]*b;
        res[3] = a[3]*b;
        res[4] = a[4]*b;
        res[5] = a[5]*b;
        res[6] = a[6]*b;
        res[7] = a[7]*b;
        res[8] = a[8]*b;
        res[9] = a[9]*b;
        res[10] = a[10]*b;
        res[11] = a[11]*b;
        res[12] = a[12]*b;
        res[13] = a[13]*b;
        res[14] = a[14]*b;
        res[15] = a[15]*b;
        res
    }
    }

// sadd
// scalar/multivector addition
impl Add<MVec> for Scalar {
    type Output = MVec;

    fn add(self: Scalar, b: MVec) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0] = a+b[0];
        res[1] = b[1];
        res[2] = b[2];
        res[3] = b[3];
        res[4] = b[4];
        res[5] = b[5];
        res[6] = b[6];
        res[7] = b[7];
        res[8] = b[8];
        res[9] = b[9];
        res[10] = b[10];
        res[11] = b[11];
        res[12] = b[12];
        res[13] = b[13];
        res[14] = b[14];
        res[15] = b[15];
        res
    }
}

// adds
// multivector/scalar addition
impl Add<Scalar> for MVec {
    type Output = MVec;

    fn add(self: MVec, b: Scalar) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0] = a[0]+b;
        res[1] = a[1];
        res[2] = a[2];
        res[3] = a[3];
        res[4] = a[4];
        res[5] = a[5];
        res[6] = a[6];
        res[7] = a[7];
        res[8] = a[8];
        res[9] = a[9];
        res[10] = a[10];
        res[11] = a[11];
        res[12] = a[12];
        res[13] = a[13];
        res[14] = a[14];
        res[15] = a[15];
        res
    }
    }

// ssub
// scalar/multivector subtraction
impl Sub<MVec> for Scalar {
    type Output = MVec;

    fn sub(self: Scalar, b: MVec) -> MVec {
        let mut res = MVec::zero();
        let a = self;
        res[0] = a-b[0];
        res[1] = -b[1];
        res[2] = -b[2];
        res[3] = -b[3];
        res[4] = -b[4];
        res[5] = -b[5];
        res[6] = -b[6];
        res[7] = -b[7];
        res[8] = -b[8];
        res[9] = -b[9];
        res[10] = -b[10];
        res[11] = -b[11];
        res[12] = -b[12];
        res[13] = -b[13];
        res[14] = -b[14];
        res[15] = -b[15];
        res
    }
}

impl MVec {
    pub fn norm(&self) -> Scalar {
        let scalar_part = (*self * self.Conjugate())[0];

        scalar_part.abs().sqrt()
    }

    pub fn inorm(&self) -> Scalar {
        self.dual().norm()
    }

    pub fn normalized(&self) -> Self {
        *self * (1.0 / self.norm())
    }
    
    
    // A rotor (Euclidean line) and translator (Ideal line)
    pub fn rotor(angle: Scalar, line: Self) -> Self {
        (angle / 2.0).cos() + (angle / 2.0).sin() * line.normalized()
    }

    pub fn translator(dist: Scalar, line: Self) -> Self {
        1.0 + dist / 2.0 * line
    }

    // A plane is defined using its homogenous equation ax + by + cz + d = 0
    pub fn plane(a: Scalar, b: Scalar, c:Scalar, d:Scalar) -> Self {
        //a * e1 + b * e2 + c * e3 + d * e0
        let mut res = MVec::zero();
        res[e1i] = a;
        res[e2i] = b;
        res[e3i] = c;
        res[e0i] = d;
        res
    }

    // PGA lines are bivectors.
    pub fn e01() -> Self { e0^e1 } 
    pub fn e02() -> Self { e0^e2 }
    pub fn e03() -> Self { e0^e3 }
    pub fn e12() -> Self { e1^e2 } 
    pub fn e31() -> Self { e3^e1 }
    pub fn e23() -> Self { e2^e3 }

    // PGA points are trivectors.
    pub fn e123() -> Self { e1^e2^e3 }
    pub fn e032() -> Self { e0^e3^e2 }
    pub fn e013() -> Self { e0^e1^e3 }
    pub fn e021() -> Self { e0^e2^e1 }

    // A point is just a homogeneous point, euclidean coordinates plus the origin
    pub fn point(x: Scalar, y: Scalar, z:Scalar) -> Self {
         //Self::e123() + x * Self::e032() + y * Self::e013() + z * Self::e021()
         let mut res = MVec::zero();
         res[e123i] = 1.;
         res[e032i] = x;
         res[e013i] = y;
         res[e021i] = z;
         res
    }

    // for our toy problem (generate points on the surface of a torus)
    // we start with a function that generates motors.
    // circle(t) with t going from 0 to 1.
    pub fn circle(t: Scalar, radius: Scalar, line: Self) -> Self {
        Self::rotor(t * 2.0 * PI, line) * Self::translator(radius, e1 * e0)
    }

    // a torus is now the product of two circles.
    pub fn torus(s: Scalar, t: Scalar, r1: Scalar, l1: Self, r2: Scalar, l2: Self) -> Self {
        Self::circle(s, r2, l2) * Self::circle(t, r1, l1)
    }

    // and to sample its points we simply sandwich the origin ..
    pub fn point_on_torus(s: Scalar, t: Scalar) -> Self {
        let to: Self = Self::torus(s, t, 0.25, Self::e12(), 0.6, Self::e31());

        to * Self::e123() * to.Reverse()
    }

}

impl From<Point> for MVec {
    fn from(point: Point) -> Self {
        let mut res = MVec::zero();
        res[e032i] = point[0];
        res[e013i] = point[1];
        res[e021i] = point[2];
        res[e123i] = point[3];
        res
    }
}
impl From<Line> for MVec {
    fn from(line: Line) -> Self {
        let mut res = MVec::zero();
        res[e01i] = line[0];
        res[e02i] = line[1];
        res[e03i] = line[2];
        res[e12i] = line[3];
        res[e31i] = line[4];
        res[e23i] = line[5];
        res
    }
}
impl From<Plane> for MVec {
    fn from(plane: Plane) -> Self {
        let mut res = MVec::zero();
        res[e01i] = plane[0];
        res[e02i] = plane[1];
        res[e03i] = plane[2];
        res[e12i] = plane[3];
        res[e31i] = plane[4];
        res[e23i] = plane[5];
        res
    }
}

struct Point {
    /// x*e032 + y*e013 + z*e021 + w?*e123
    /// [x,y,z,w?]
    pub terms: [Scalar; 4]
}
impl Index<usize> for Point {
    type Output = Scalar;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.terms[index]
    }
}
impl IndexMut<usize> for Point {
    
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.terms[index]
    }
}

struct Line {
    /// a*e01 + b*e02 + c*e03 + d*e12 + e*e31 + f*e23
    pub terms: [Scalar; 6]
}
impl Index<usize> for Line {
    type Output = Scalar;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.terms[index]
    }
}
impl IndexMut<usize> for Line {
    
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.terms[index]
    }
}

struct Plane {
    /// a*e1 + b*e2 + c*e3 + d*e0
    pub terms: [Scalar; 4]
}

impl Index<usize> for Plane {
    type Output = Scalar;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.terms[index]
    }
}
impl IndexMut<usize> for Plane {
    
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.terms[index]
    }
}

struct Mover {
    pub radius: Scalar,
    /// KEEP THIS AS A EUCLIDIAN (non ideal) POINT
    pub center: MVec,

    /// keep this as an IDEAL point
    pub facing: MVec,

    /// bivector = line?
    pub velocity: MVec,

    /// bivector = line?
    pub momentum: MVec,
}

fn main() {
    println!("Hello, world!");
}