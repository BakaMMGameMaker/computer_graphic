#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

// 钳制到 [0, 1]
static inline float clamp01(float x) { return std::min(1.0f, std::max(0.0f, x)); }

struct Vec3 {
    float x{}, y{}, z{};
    Vec3() = default;
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    Vec3 operator+(const Vec3 &r) const { return {x + r.x, y + r.y, z + r.z}; }
    Vec3 operator-(const Vec3 &r) const { return {x - r.x, y - r.y, z - r.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
};

// w = 1: 点，w = 0: 向量
struct Vec4 {
    float x{}, y{}, z{}, w{};
    Vec4() = default;
    Vec4(float X, float Y, float Z, float W) : x(X), y(Y), z(Z), w(W) {}
};

// 点乘
static inline float dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// 叉积
static inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// 模长
static inline float length(const Vec3 &v) { return std::sqrt(dot(v, v)); }

// 归一化
static inline Vec3 normalize(const Vec3 &v) {
    float len = length(v);
    if (len <= 1e-20f) return {0, 0, 0};
    return v / len;
}

// 齐次坐标
struct Mat4 {
    // row-major 4x4
    float m[4][4]{};

    // 单位矩阵
    static Mat4 identity() {
        Mat4 r;
        for (int i = 0; i < 4; i++) r.m[i][i] = 1.0f;
        return r;
    }
};

// 应用齐次坐标变换
static inline Vec4 mul(const Mat4 &A, const Vec4 &v) {
    Vec4 r;
    r.x = A.m[0][0] * v.x + A.m[0][1] * v.y + A.m[0][2] * v.z + A.m[0][3] * v.w;
    r.y = A.m[1][0] * v.x + A.m[1][1] * v.y + A.m[1][2] * v.z + A.m[1][3] * v.w;
    r.z = A.m[2][0] * v.x + A.m[2][1] * v.y + A.m[2][2] * v.z + A.m[2][3] * v.w;
    r.w = A.m[3][0] * v.x + A.m[3][1] * v.y + A.m[3][2] * v.z + A.m[3][3] * v.w;
    return r;
}

// 齐次坐标矩阵乘积
static inline Mat4 mul(const Mat4 &A, const Mat4 &B) {
    Mat4 R{};
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += A.m[r][k] * B.m[k][c];
            R.m[r][c] = s;
        }
    }
    return R;
}

// note：右手坐标系，y 方向为 up 方向，相机看向 -z 方向
// eye：相机在世界系中的位置
// center：相机看向的目标点
// up：世界的向上方向
// return；返回将世界系转换为相机系的齐次变换矩阵
static Mat4 lookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up) {
    Vec3 f = normalize(center - eye); // 相机 to 目标点方向
    Vec3 s = normalize(cross(f, up)); // 相机的右方向
    Vec3 u = cross(s, f);             // 相机的向上方向

    // 构建齐次坐标
    Mat4 M = Mat4::identity();
    // 构建左上角 3x3 旋转部分
    M.m[0][0] = s.x;
    M.m[0][1] = s.y;
    M.m[0][2] = s.z;
    M.m[1][0] = u.x;
    M.m[1][1] = u.y;
    M.m[1][2] = u.z;
    M.m[2][0] = -f.x;
    M.m[2][1] = -f.y;
    M.m[2][2] = -f.z;
    // 构建最后一列平移部分，不能直接 eye.，因为是先平移再旋转
    // 这里直接给出了复合结果
    M.m[0][3] = -dot(s, eye);
    M.m[1][3] = -dot(u, eye);
    M.m[2][3] = dot(f, eye);
    return M; // 应用给场景中的物体、顶点、法线等
}

// fovyRadians：垂直视角
// aspect：画面宽高比
// zNear：近面离相机的距离
// zFar：远面离相机的距离
static Mat4 perspective(float fovyRadians, float aspect, float zNear, float zFar) {
    // tan = 近面半高 / zNear
    float f = 1.0f / std::tan(fovyRadians * 0.5f); // zNear / 近面半高
    Mat4 P{};
    // ？
    P.m[0][0] = f / aspect;
    P.m[1][1] = f;
    P.m[2][2] = (zFar + zNear) / (zNear - zFar);
    P.m[2][3] = (2.0f * zFar * zNear) / (zNear - zFar);
    P.m[3][2] = -1.0f;
    return P;
}

// 顶点信息
struct Vertex {
    Vec3 pos;    // 顶点在世界的位置 set model=identity
    Vec3 normal; // 顶点在世界坐标系下的法向量
    Vec3 kd;     // 漫反射颜色 RGB [0-1]
};

struct Varying {
    Vec3 worldPos; // 世界系位置
    Vec3 normal;   // 世界系法向
    Vec3 kd;       // 漫反射颜色
    float invW;    // 1/clip.w：透视正确插值？
    Vec3 ndc;      // 标准化坐标？
    Vec3 screen;   // 像素坐标与深度值，用于光栅化，判断像素是否在三角形内和深度测试
};

// 判断像素是否在三角形内部，同时计算重心坐标
// 重心坐标：点对某边的小三角形面积 / 整三角形面积 = 对面顶点的权重
// return：三角形某边和该顶点形成的小三角形的面积的两倍
static inline float edge(const Vec3 &a, const Vec3 &b, float x, float y) {
    return (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
}

static void writePPM(const char *path, int W, int H, const std::vector<Vec3> &fb) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            size_t idx = static_cast<size_t>(y * W + x);
            const Vec3 &c = fb[idx];
            uint8_t r = static_cast<uint8_t>(std::lround(255.0f * clamp01(c.x)));
            uint8_t g = static_cast<uint8_t>(std::lround(255.0f * clamp01(c.y)));
            uint8_t b = static_cast<uint8_t>(std::lround(255.0f * clamp01(c.z)));
            out.write(reinterpret_cast<const char *>(&r), 1);
            out.write(reinterpret_cast<const char *>(&g), 1);
            out.write(reinterpret_cast<const char *>(&b), 1);
        }
    }
}

int main() {
    // 定义屏幕宽高
    const int W = 800;
    const int H = 600;

    std::vector<Vec3> framebuffer(W * H, Vec3(0.02f, 0.02f, 0.03f)); // 背景色
    std::vector<float> depthbuffer(W * H, std::numeric_limits<float>::infinity());

    // 定义一个三角形
    std::array<Vertex, 3> tri = {Vertex{Vec3(-0.8f, -0.6f, 0.0f), Vec3(0, 0, 1), Vec3(0.9f, 0.2f, 0.2f)},
                                 Vertex{Vec3(0.8f, -0.6f, 0.0f), Vec3(0, 0, 1), Vec3(0.2f, 0.9f, 0.2f)},
                                 Vertex{Vec3(0.0f, 0.8f, 0.0f), Vec3(0, 0, 1), Vec3(0.2f, 0.2f, 0.9f)}};

    // 定义相机位置，目标点和世界向上方向
    Vec3 eye(0, 0, 2.5f);
    Vec3 center(0, 0, 0);
    Vec3 up(0, 1, 0);

    Mat4 V = lookAt(eye, center, up);
    Mat4 P = perspective(60.0f * 3.1415926f / 180.0f, float(W) / float(H), 0.1f, 50.0f);
    Mat4 VP = mul(P, V);

    Vec3 lightPos(1.0f, 1.5f, 2.0f);  // 光源位置
    Vec3 lightI(25.0f, 25.0f, 25.0f); // 光源强度

    std::array<Varying, 3> v{};
    for (size_t i = 0; i < 3; i++) {
        Vec3 wp = tri[i].pos;                              // 当前顶点的世界坐标
        Vec4 clip = mul(VP, Vec4(wp.x, wp.y, wp.z, 1.0f)); // 为顶点应用变换矩阵
        float invW = 1.0f / clip.w;                        // clip坐标？用于透视正确插值

        // 执行透视除法，得到标准化设备坐标 NDC？
        Vec3 ndc = {clip.x * invW, clip.y * invW, clip.z * invW}; // z：[-1,1]

        // 视口变换，NDC to 屏幕
        float sx = (ndc.x * 0.5f + 0.5f) * float(W);
        float sy = (1.0f - (ndc.y * 0.5f + 0.5f)) * float(H); // 翻转 y 轴，因为图像坐标系下 y 轴向下
        float sd = ndc.z * 0.5f + 0.5f;                       // 深度：[-1,1] -> [0,1]

        // 构造 Varing
        v[i].worldPos = wp;
        v[i].normal = tri[i].normal;
        v[i].kd = tri[i].kd;
        v[i].invW = invW;
        v[i].ndc = ndc;
        v[i].screen = {sx, sy, sd};
    }

    // 光栅化
    // 三角形包围盒
    float minX = std::floor(std::min({v[0].screen.x, v[1].screen.x, v[2].screen.x}));
    float maxX = std::ceil(std::max({v[0].screen.x, v[1].screen.x, v[2].screen.x}));
    float minY = std::floor(std::min({v[0].screen.y, v[1].screen.y, v[2].screen.y}));
    float maxY = std::ceil(std::max({v[0].screen.y, v[1].screen.y, v[2].screen.y}));
    int x0 = std::max(0, static_cast<int>(minX));
    int x1 = std::min(W - 1, static_cast<int>(maxX));
    int y0 = std::max(0, static_cast<int>(minY));
    int y1 = std::min(H - 1, static_cast<int>(maxY));

    // 三角形面积的两倍
    float area = edge(v[0].screen, v[1].screen, v[2].screen.x, v[2].screen.y);

    if (std::abs(area) < 1e-10f) {
        std::cerr << "Degenerate triangle.\n";
        writePPM("out.ppm", W, H, framebuffer);
        return 0;
    }

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            // 像素中心坐标
            float px = float(x) + 0.5f;
            float py = float(y) + 0.5f;

            // 用于判断像素是否在三角形内部，并用于后续权重计算
            float w0 = edge(v[1].screen, v[2].screen, px, py);
            float w1 = edge(v[2].screen, v[0].screen, px, py);
            float w2 = edge(v[0].screen, v[1].screen, px, py);

            // 逆时针时 area > 0，顺时针时 < 0
            if ((w0 >= 0 && w1 >= 0 && w2 >= 0 && area > 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0 && area < 0)) {
                // 各顶点权重（重心坐标），sigma = 1
                float a = w0 / area;
                float b = w1 / area;
                float c = w2 / area;

                // perspective-correct weights
                float iw0 = v[0].invW, iw1 = v[1].invW, iw2 = v[2].invW;
                float denom = a * iw0 + b * iw1 + c * iw2; // 加权求和
                if (std::abs(denom) < 1e-20f) continue;

                // 透视正确的重心坐标，sigma = 1
                float A = (a * iw0) / denom;
                float B = (b * iw1) / denom;
                float C = (c * iw2) / denom;

                // 透视正确的深度
                float depth = A * v[0].screen.z + B * v[1].screen.z + C * v[2].screen.z;

                size_t idx = static_cast<size_t>(y * W + x);
                // 深度测试
                if (depth < depthbuffer[idx]) {
                    depthbuffer[idx] = depth;

                    Vec3 _P = v[0].worldPos * A + v[1].worldPos * B + v[2].worldPos * C;     // 当前像素世界空间位置
                    Vec3 N = normalize(v[0].normal * A + v[1].normal * B + v[2].normal * C); // 当前像素世界空间法线
                    Vec3 kd = v[0].kd * A + v[1].kd * B + v[2].kd * C;                       // 当前像素漫反射颜色系数

                    Vec3 L = lightPos - _P;                // 像素到点光源向量
                    float r2 = std::max(1e-6f, dot(L, L)); // 到光源距离平方
                    Vec3 ldir = normalize(L);
                    float cosTheta = std::max(0.0f, dot(N, ldir)); // 夹角余弦

                    Vec3 color = Vec3(kd.x * (lightI.x / r2) * cosTheta, kd.y * (lightI.y / r2) * cosTheta,
                                      kd.z * (lightI.z / r2) * cosTheta); // 应用漫反射模型，计算三通道各自颜色

                    framebuffer[idx] = color;
                }
            }
        }
    }

    writePPM("out.ppm", W, H, framebuffer);
    std::cout << "Wrote out.ppm\n";
    return 0;
}