#define MAX(a, b) ((a > b) ? (a) : (b))
#define MIN(a, b) ((a < b) ? (a) : (b))

typedef struct __attribute__((packed)) _args {
    float transform[6]; // affine matrix, when applied to warping, this can be derived from the
                        // mapping's jacobian (this should be done per-pixel in reality inside this
                        // kernel)
    uint in_dim[2];
    uint out_dim[2];
} kernel_args;

// __constant float gamma_ = 2.2f;

// inline float to_linear(float x) {
//     x /= 255.f;
//     if (gamma_ == 0.f) {
//         if (x <= .04045f) {
//             return x / 12.92f;
//         }
//         return pow((x + .055f) / 1.055f, 2.4f);
//     }
//     return pow(x, gamma_);
// }

// inline float to_percieved(float x) {
//     if (gamma_ == 0.f) {
//         if (x <= .0031308f) {
//             return x * 12.92f * 255.f;
//         }
//         return (1.055f * pow(x, 1 / 2.4f) - .055f) * 255.f;
//     }
//     return pow(x, 1.f / gamma_) * 255.f;
// }

void clamped_ellipse(float m0, float m1, float m2, float m3, float *abc) {
    /* find ellipse */
    const float F0 = fabs(m0 * m3 - m1 * m2);
    const float F = MAX(0.1f, F0 * F0);
    const float A = (m2 * m2 + m3 * m3) / F;
    const float B = -2 * (m0 * m2 + m1 * m3) / F;
    const float C = (m0 * m0 + m1 * m1) / F;

    /* find the angle to rotate ellipse */
    const float2 v = {C - A, -B};
    const float lv = length(v);
    const float v0 = (lv > 1e-2) ? v[0] / lv : 1.f;
    const float v1 = (lv > 1e-2) ? v[1] / lv : 1.f;

    const float c = sqrt(MAX(0, 1 + v0) / 2);
    float s = sqrt(MAX(1 - v0, 0) / 2);

    /* rotate the ellipse to align it with axes */
    float A0 = (A * c * c - B * c * s + C * s * s);
    float C0 = (A * s * s + B * c * s + C * c * c);

    const float Bt1 = B * (c * c - s * s);
    const float Bt2 = 2 * (A - C) * c * s;
    float B0 = Bt1 + Bt2;
    const float B0v2 = Bt1 - Bt2;
    if (fabs(B0) > fabs(B0v2)) {
        s = -s;
        B0 = B0v2;
    }

    /* clip A,C */
    A0 = MIN(A0, 1);
    C0 = MIN(C0, 1);

    const float sn = -s;

    /* rotate it back */
    abc[0] = (A0 * c * c - B0 * c * sn + C0 * sn * sn);
    abc[1] = (2 * A0 * c * sn + B0 * c * c - B0 * sn * sn - 2 * C0 * c * sn);
    abc[2] = (A0 * sn * sn + B0 * c * sn + C0 * c * c);
}

// Keys Cubic Filter Family
// https://imagemagick.org/Usage/filter/#robidoux
inline float bc2(float x) {
    const float B = .3782; // Robidoux filter
    const float C = (1.f - B) / 2.;
    const float P0 = (6.f - 2.f * B) / 6.f;
    const float P1 = 0.;
    const float P2 = (-18.f + 12.f * B + 6.f * C) / 6.f;
    const float P3 = (12.f - 9.f * B - 6.f * C) / 6.f;
    const float Q0 = (8.f * B + 24.f * C) / 6.f;
    const float Q1 = (-12.f * B - 48.f * C) / 6.f;
    const float Q2 = (6.f * B + 30.f * C) / 6.f;
    const float Q3 = (-1.f * B - 6.f * C) / 6.f;
    if (x < 0)
        x = -x;
    if (x < 1.f)
        return P0 + P1 * x + P2 * x * x + P3 * x * x * x;
    if (x < 2.f)
        return Q0 + Q1 * x + Q2 * x * x + Q3 * x * x * x;
    return 0.;
}

inline void affine_transform(const float m[6], const float in[2], float out[2]) {
    out[0] = m[0] * in[0] + m[1] * in[1] + m[2];
    out[1] = m[3] * in[0] + m[4] * in[1] + m[5];
}

inline void affine_bbox(const float in[6], float out[2]) {
    out[0] = 2 * max(fabs(in[0] + in[1]), fabs(in[0] - in[1]));
    out[1] = 2 * max(fabs(in[3] + in[4]), fabs(in[3] - in[4]));
}

// float2 trans(float x, float y) {
//     float cx = 1920 / 2., cy = 1080 / 2.;
//     x -= cx;
//     y -= cy;
//     float r2 = sqrt(x * x + y * y);
//     // if (r2 / 1000 > 1) {
//     //     float2 r = {-99999, -99999};
//     //     return r;
//     // }
//     float t = atan(r2 / 2000);
//     // float t = sin(r2 / 1000);
//     float k = 1 - 1*t;
//     x *= k * 2;
//     y *= k * 2;
//     x += cx;
//     y += cy;
//     float2 r = {x / 1920 * 4000, y / 1080 * 3000};
//     // float2 r = {x * 1 * (x + 200) * (1 / 1300.),
//     // y * 2 * (y + 1500) * (1 / 2000.) + 50 * sin(x / 100)};
//     return r;
// }

float2 trans(float x, float y) {
    float2 r = {x * 1 * (x + 200) * (1 / 1300.),
                y * 2 * (y + 1500) * (1 / 2000.) + 50 * sin(x / 100)};
    return r;
}

kernel void warp(global const float *in_image, global float *out_image, kernel_args args) {
    uint i = get_global_id(0);
    uint j = get_global_id(1);

    float2 uv = trans(j, i);
    float2 dx = trans(j + 1e-2f, i) - uv;
    float2 dy = trans(j, i + 1e-2f) - uv;
    float dudx = dx.x / 1e-2f;
    float dudy = dy.x / 1e-2f;
    float dvdx = dx.y / 1e-2f;
    float dvdy = dy.y / 1e-2f;
    float det = dudx * dvdy - dvdx * dudy;

    args.transform[0] = dudx;
    args.transform[1] = dudy;
    args.transform[2] = (uv.x - j * dudx - i * dudy);
    args.transform[3] = dvdx;
    args.transform[4] = dvdy;
    args.transform[5] = (uv.y - j * dvdx - i * dvdy);

    /* find where the pixel is on source image */
    float cntr_in[2];
    float cntr_out[2] = {j, i};
    affine_transform(args.transform, cntr_out, cntr_in);
    int cntr_int[2] = {round(cntr_in[0]), round(cntr_in[1])};
    float cntr_frac[2] = {cntr_in[0] - cntr_int[0], cntr_in[1] - cntr_int[1]};

    /* find how many pixels we need around that pixel in each direction */
    float trans_size[2];
    affine_bbox(args.transform, trans_size);

    /* find bounding box of samling region in the input image */
    int own_minx = MAX(0, floor(MIN(cntr_in[0] - trans_size[0] - 1, cntr_in[0] - 2))) - cntr_int[0];
    int own_miny = MAX(0, floor(MIN(cntr_in[1] - trans_size[1] - 1, cntr_in[1] - 2))) - cntr_int[1];
    int own_maxx =
        MIN(args.in_dim[1] - 1, ceil(MAX(cntr_in[0] + trans_size[0] + 1, cntr_in[0] + 2))) -
        cntr_int[0];
    int own_maxy =
        MIN(args.in_dim[1] - 1, ceil(MAX(cntr_in[0] + trans_size[0] + 1, cntr_in[0] + 2))) -
        cntr_int[0];

    int in_x, in_y;
    float3 sum = 0;
    float sum_div = 0;
    // See: Andreas Gustafsson. "Interactive Image Warping", section 3.6
    // http://www.gson.org/thesis/warping-thesis.pdf
    float abc[3];
    clamped_ellipse(args.transform[0], args.transform[1], args.transform[3], args.transform[4],
                    abc);
    for (in_y = own_miny; in_y <= own_maxy; ++in_y) {
        const float in_fy = in_y - cntr_frac[1];
#pragma unroll
        for (in_x = own_minx; in_x <= own_maxx; ++in_x) {
            const float in_fx = in_x - cntr_frac[0];
            const float dr =
                in_fx * in_fx * abc[0] + in_fx * in_fy * abc[1] + in_fy * in_fy * abc[2];
            const float k = bc2(sqrt(dr)); // cylindrical filtering
            if (k == 0)
                continue;
            const int p = 3 * ((in_y + cntr_int[1]) * args.in_dim[1] + in_x + cntr_int[0]);
            const float3 smpl = {in_image[p + 0], in_image[p + 1], in_image[p + 2]};
            sum += k * smpl;
            sum_div += k;
        }
    }
    if (i < args.out_dim[0] && j < args.out_dim[1] && cntr_in[0] < args.in_dim[1] &&
        cntr_in[1] < args.in_dim[0] && cntr_in[0] >= 0 && cntr_in[1] >= 0) {
        const int p = 3 * (args.out_dim[1] * i + j);
        const float3 v = (sum / sum_div);
        out_image[p + 0] = v[0];
        out_image[p + 1] = v[1];
        out_image[p + 2] = v[2];
    }
}