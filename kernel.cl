__constant float lanczos3_table128[128] = {
    1.00000000e+00,  9.98980463e-01,  9.95926142e-01,  9.90849495e-01,  9.83771443e-01,
    9.74720955e-01,  9.63735044e-01,  9.50858653e-01,  9.36144233e-01,  9.19651628e-01,
    9.01447535e-01,  8.81605446e-01,  8.60205054e-01,  8.37331951e-01,  8.13077092e-01,
    7.87536383e-01,  7.60810137e-01,  7.33002663e-01,  7.04221606e-01,  6.74577415e-01,
    6.44182920e-01,  6.13152564e-01,  5.81601977e-01,  5.49647391e-01,  5.17405033e-01,
    4.84990507e-01,  4.52518344e-01,  4.20101404e-01,  3.87850344e-01,  3.55873138e-01,
    3.24274540e-01,  2.93155670e-01,  2.62613475e-01,  2.32740477e-01,  2.03624249e-01,
    1.75347075e-01,  1.47985741e-01,  1.21611148e-01,  9.62880775e-02,  7.20750317e-02,
    4.90240380e-02,  2.71805003e-02,  6.58313977e-03,  -1.27360839e-02, -3.07519734e-02,
    -4.74460721e-02, -6.28066137e-02, -7.68283978e-02, -8.95126835e-02, -1.00867018e-01,
    -1.10905014e-01, -1.19646154e-01, -1.27115503e-01, -1.33343428e-01, -1.38365313e-01,
    -1.42221183e-01, -1.44955382e-01, -1.46616206e-01, -1.47255510e-01, -1.46928310e-01,
    -1.45692378e-01, -1.43607855e-01, -1.40736818e-01, -1.37142822e-01, -1.32890597e-01,
    -1.28045514e-01, -1.22673288e-01, -1.16839513e-01, -1.10609308e-01, -1.04046948e-01,
    -9.72155184e-02, -9.01765674e-02, -8.29897895e-02, -7.57127479e-02, -6.84005991e-02,
    -6.11058325e-02, -5.38780540e-02, -4.67637926e-02, -3.98063287e-02, -3.30455415e-02,
    -2.65178010e-02, -2.02558748e-02, -1.42888604e-02, -8.64215754e-03, -3.33744590e-03,
    1.60728837e-03,  6.17772061e-03,  1.03631206e-02,  1.41562661e-02,  1.75533369e-02,
    2.05537844e-02,  2.31601894e-02,  2.53780913e-02,  2.72158198e-02,  2.86842901e-02,
    2.97968090e-02,  3.05688586e-02,  3.10178660e-02,  3.11629865e-02,  3.10248621e-02,
    3.06253918e-02,  2.99874917e-02,  2.91348584e-02,  2.80917436e-02,  2.68827174e-02,
    2.55324580e-02,  2.40655281e-02,  2.25061718e-02,  2.08781250e-02,  1.92044266e-02,
    1.75072532e-02,  1.58077590e-02,  1.41259311e-02,  1.24804694e-02,  1.08886659e-02,
    9.36631579e-03,  7.92763475e-03,  6.58519613e-03,  5.34988381e-03,  4.23086155e-03,
    3.23556084e-03,  2.36967835e-03,  1.63719582e-03,  1.04040874e-03,  5.79971762e-04,
    2.54957093e-04,  6.29239439e-05,  0.00000000e+00};

__constant float lanczos3_table32[32] = {
    1.,         0.98298126,  0.93308896,  0.85371226,  0.7501684,   0.62924045,  0.49859726,
    0.366152,   0.2394229,   0.12495654,  0.02786587,  -0.04847741, -0.10257464, -0.13475333,
    -0.1469572, -0.14240487, -0.12516053, -0.09966648, -0.070288,   -0.0409162,  -0.01466442,
    0.00631879, 0.02091215,  0.0289661,   0.03114835,  0.0287192,   0.02327073,  0.0164646,
    0.00980027, 0.004438,    0.0010922,   0.};

typedef struct __attribute__((packed)) _args {
    float transform[6];
    uint in_dim[2];
    uint out_dim[2];
} kernel_args;

inline float lanczos3(float x) {
    if (x < 0)
        x = -x;
    if (x > 3)
        return 0.0f;
    return lanczos3_table128[(uint)round(x / 3.f * 127.f)];
    // return lanczos3_table32[(uint)round(x / 3.f * 31.f)];
}

__constant float gamma_ = 2.2f;

inline float to_linear(float x) {
    x /= 255.f;
    if (gamma_ == 0.f) {
        if (x <= .04045f) {
            return x / 12.92f;
        }
        return pow((x + .055f) / 1.055f, 2.4f);
    }
    return pow(x, gamma_);
}
inline float to_percieved(float x) {
    if (gamma_ == 0.f) {
        if (x <= .0031308f) {
            return x * 12.92f * 255.f;
        }
        return (1.055f * pow(x, 1 / 2.4f) - .055f) * 255.f;
    }
    return pow(x, 1.f / gamma_) * 255.f;
}

inline float bc2(float x) {
    const float B = .3782;
    const float C = (1. - B) / 2.;
    // const float C = .5;
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

void invert_affine(const float in[6], float out[6]) {
    float det = in[0] * in[4] - in[1] * in[3];
    out[0] = in[4] / det;
    out[4] = in[0] / det;
    out[1] = -in[1] / det;
    out[3] = -in[3] / det;
    out[2] = -out[0] * in[2] - out[1] * in[5];
    out[5] = -out[3] * in[2] - out[4] * in[5];
}

void affine_bbox(const float in[6], float out[2]) {
    out[0] = 3 * max(fabs(in[0] + in[1]), fabs(in[0] - in[1]));
    out[1] = 3 * max(fabs(in[3] + in[4]), fabs(in[3] - in[4]));
}

#define MAX(a, b) ((a > b) ? (a) : (b))
#define MIN(a, b) ((a < b) ? (a) : (b))

kernel void add(global const float *in_image, global float *out_image, kernel_args args) {
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    uint li = get_local_id(0);
    uint lj = get_local_id(1);

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

    /* find bounding box */
    const int own_minx = MAX(0, floor(cntr_in[0] - trans_size[0] - 1)) - cntr_int[0];
    const int own_miny = MAX(0, floor(cntr_in[1] - trans_size[1] - 1)) - cntr_int[1];
    const int own_maxx =
        MIN(args.in_dim[1] - 1, ceil(cntr_in[0] + trans_size[0] + 1)) - cntr_int[0];
    const int own_maxy =
        MIN(args.in_dim[0] - 1, ceil(cntr_in[1] + trans_size[1] + 1)) - cntr_int[1];

    int in_x, in_y;
    float3 sum = 0;
    float sum_div = 0;
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
            float dr = in_fx * in_fx * A + in_fx * in_fy * B + in_fy * in_fy * C;
            float k = bc2(sqrt(dr));
            if (k == 0)
                continue;
            const int p = 3*((in_y + cntr_int[1]) * args.in_dim[1] + in_x + cntr_int[0]);
            const float3 smpl = {in_image[p + 0], in_image[p + 1], in_image[p + 2]};
            sum += k * smpl;
            sum_div += k;
        }
    }
    if (i < args.out_dim[0] && j < args.out_dim[1]) {
        const int p =3*(args.out_dim[1] * i + j);
        const float3 v = (sum / sum_div);
        out_image[p + 0] = v[0];
        out_image[p + 1] = v[1];
        out_image[p + 2] = v[2];
    }
}