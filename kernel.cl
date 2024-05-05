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

// Keys Cubic Filter Family
// https://imagemagick.org/Usage/filter/#robidoux
inline float bc2(float x) {
    const float B = .3782; // Robidoux filter
    const float C = (1. - B) / 2.;
    const float P0 = (6. - 2. * B) / 6.;
    const float P1 = 0.;
    const float P2 = (-18. + 12. * B + 6. * C) / 6.;
    const float P3 = (12. - 9. * B - 6. * C) / 6.;
    const float Q0 = (8. * B + 24. * C) / 6.;
    const float Q1 = (-12. * B - 48. * C) / 6.;
    const float Q2 = (6. * B + 30. * C) / 6.;
    const float Q3 = (-1. * B - 6. * C) / 6.;
    if (x < 0)
        x = -x;
    if (x < 1.)
        return P0 + P1 * x + P2 * x * x + P3 * x * x * x;
    if (x < 2.)
        return Q0 + Q1 * x + Q2 * x * x + Q3 * x * x * x;
    return 0.;
}

inline void affine_transform(const float m[6], const float in[2], float out[2]) {
    out[0] = m[0] * in[0] + m[1] * in[1] + m[2];
    out[1] = m[3] * in[0] + m[4] * in[1] + m[5];
}

inline void invert_affine(const float in[6], float out[6]) {
    float det = in[0] * in[4] - in[1] * in[3];
    out[0] = in[4] / det;
    out[4] = in[0] / det;
    out[1] = -in[1] / det;
    out[3] = -in[3] / det;
    out[2] = -out[0] * in[2] - out[1] * in[5];
    out[5] = -out[3] * in[2] - out[4] * in[5];
}

inline void affine_bbox(const float in[6], float out[2]) {
    out[0] = 2 * max(fabs(in[0] + in[1]), fabs(in[0] - in[1]));
    out[1] = 2 * max(fabs(in[3] + in[4]), fabs(in[3] - in[4]));
}

#define MAX(a, b) ((a > b) ? (a) : (b))
#define MIN(a, b) ((a < b) ? (a) : (b))

kernel void warp(global const float *in_image, global float *out_image, kernel_args args) {
    uint i = get_global_id(0);
    uint j = get_global_id(1);

    /* find where the pixel is on source image */
    float cntr_in[2];
    float cntr_out[2] = {j, i};
    affine_transform(args.transform, cntr_out, cntr_in);
    int cntr_int[2] = {round(cntr_in[0]), round(cntr_in[1])};
    float cntr_frac[2] = {cntr_in[0] - cntr_int[0], cntr_in[1] - cntr_int[1]};
    float transform_inv[6];
    invert_affine(args.transform, transform_inv);

    /* find how many pixels we need around that pixel in each direction */
    float trans_size[2];
    affine_bbox(args.transform, trans_size);

    /* find bounding box of samling region in the input image */
    const int own_minx = MAX(0, floor(cntr_in[0] - trans_size[0] - 1)) - cntr_int[0];
    const int own_miny = MAX(0, floor(cntr_in[1] - trans_size[1] - 1)) - cntr_int[1];
    const int own_maxx =
        MIN(args.in_dim[1] - 1, ceil(cntr_in[0] + trans_size[0] + 1)) - cntr_int[0];
    const int own_maxy =
        MIN(args.in_dim[0] - 1, ceil(cntr_in[1] + trans_size[1] + 1)) - cntr_int[1];

    int in_x, in_y;
    float3 sum = 0;
    float sum_div = 0;
    // See: Andreas Gustafsson. "Interactive Image Warping", section 3.6
    // http://www.gson.org/thesis/warping-thesis.pdf
    const float A =
        MIN(1, transform_inv[0] * transform_inv[0] + transform_inv[3] * transform_inv[3]);
    const float B = 2 * (transform_inv[0] * transform_inv[1] + transform_inv[3] * transform_inv[4]);
    const float C =
        MIN(1, transform_inv[1] * transform_inv[1] + transform_inv[4] * transform_inv[4]);
    for (in_y = own_miny; in_y <= own_maxy; ++in_y) {
        const float in_fy = in_y - cntr_frac[1];
        const float oxy0[2] = {transform_inv[1] * in_y, transform_inv[4] * in_y};
#pragma unroll
        for (in_x = own_minx; in_x <= own_maxx; ++in_x) {
            const float in_fx = in_x - cntr_frac[0];
            const float dr = in_fx * in_fx * A + in_fx * in_fy * B + in_fy * in_fy * C;
            const float k = bc2(sqrt(dr)); // cylindrical filtering
            if (k == 0)
                continue;
            const int p = 3 * ((in_y + cntr_int[1]) * args.in_dim[1] + in_x + cntr_int[0]);
            const float3 smpl = {in_image[p + 0], in_image[p + 1], in_image[p + 2]};
            sum += k * smpl;
            sum_div += k;
        }
    }
    if (i < args.out_dim[0] && j < args.out_dim[1]) {
        const int p = 3 * (args.out_dim[1] * i + j);
        const float3 v = (sum / sum_div);
        out_image[p + 0] = v[0];
        out_image[p + 1] = v[1];
        out_image[p + 2] = v[2];
    }
}