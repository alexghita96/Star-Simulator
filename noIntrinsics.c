/**
 * i7-3612qm
 * 32KB L1 cache size
 * 64 lines
 * approximately 16 32B words per line
 */
void compute() {

  double t0, t1;

  // Loop 0.
  t0 = wtime();
  for (int i = 0; i < N; i++) {
    ax[i] = 0.0f;
  }
  for (int i = 0; i < N; i++) {
    ay[i] = 0.0f;
  }
  for (int i = 0; i < N; i++) {
    az[i] = 0.0f;
  }
  t1 = wtime();
  l0 += (t1 - t0);

  // Loop 1.
  t0 = wtime();

  float* rx = (float*) _mm_malloc(N * sizeof(float), 16);
  float* ry = (float*) _mm_malloc(N * sizeof(float), 16);
  float* rz = (float*) _mm_malloc(N * sizeof(float), 16);
  float* r2 = (float*) _mm_malloc(N * sizeof(float), 16);
  float* r2inv = (float*) _mm_malloc(N * sizeof(float), 16);
  float* r6inv = (float*) _mm_malloc(N * sizeof(float), 16);
  float* s = (float*) _mm_malloc(N * sizeof(float), 16);

  for (int i = 0; i < N; i++) {
    // prolog
    for (int j = 0; j < 5; j++) {
      rx[j] = x[j] - x[i];
      ry[j] = y[j] - y[i];
      rz[j] = z[j] - z[i];
      if (j > 0) {
        r2[j - 1] = rx[j - 1] * rx[j - 1] + ry[j - 1] * ry[j - 1] +
                    rz[j - 1] * rz[j - 1] + eps;
        if (j > 1) {
          r2inv[j - 2] = 1.0f / sqrt(r2[j - 2]);
          if (j > 2) {
            r6inv[j - 3] = r2inv[j - 3] * r2inv[j - 3] * r2inv[j - 3];
            if (j > 3) {
              s[j - 4] = m[j - 4] * r6inv[j - 4];
            }
          }
        }
      }
    }

    // pipeline
    for (int j = 5; j < N; j++) {
      rx[j] = x[j] - x[i];
      ry[j] = y[j] - y[i];
      rz[j] = z[j] - z[i];

      r2[j - 1] = rx[j - 1] * rx[j - 1] + ry[j - 1] * ry[j - 1] +
                  rz[j - 1] * rz[j - 1] + eps;

      r2inv[j - 2] = 1.0f / sqrt(r2[j - 2]);

      r6inv[j - 3] = r2inv[j - 3] * r2inv[j - 3] * r2inv[j - 3];

      s[j - 4] = m[j - 4] * r6inv[j - 4];

      ax[i] += s[j - 5] * rx[j - 5];
      ay[i] += s[j - 5] * ry[j - 5];
      az[i] += s[j - 5] * rz[j - 5];
    }

    // epilog
    for (int j = N - 5; j < N; j++) {
      if (j < N - 4) {
        r2[j + 4] = rx[j + 4] * rx[j + 4] + ry[j + 4] * ry[j + 4] +
                    rz[j + 4] * rz[j + 4] + eps;
      }
      if (j < N - 3) {
        r2inv[j + 3] = 1.0f / sqrt(r2[j + 3]);
      }
      if (j < N - 2) {
        r6inv[j + 2] = r2inv[j + 2] * r2inv[j + 2] * r2inv[j + 2];
      }
      if (j < N - 1) {
        s[j + 1] = m[j + 1] * r6inv[j + 1];
      }
      ax[i] += s[j] * rx[j];
      ay[i] += s[j] * ry[j];
      az[i] += s[j] * rz[j];
    }
  }
  t1 = wtime();
  l1 += (t1 - t0);

  // Loop 2.
  t0 = wtime();
  t1 = wtime();
  l2 += (t1 - t0);

  // Loop 3.
  // this isn't really helping
  t0 = wtime();
  for (int i = 0; i < N; i++) {
    vx[i] += dmp * (dt * ax[i]);
    x[i] += dt * vx[i];
    if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
  }
  for (int i = 0; i < N; i++) {
    vy[i] += dmp * (dt * ay[i]);
    y[i] += dt * vy[i];
    if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
  }
  for (int i = 0; i < N; i++) {
    vz[i] += dmp * (dt * az[i]);
    z[i] += dt * vz[i];
    if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;
  }
  t1 = wtime();
  l3 += (t1 - t0);
}

// for (i = 0; i < N; i++) {
//   for (j = 0; j < unroll_N; j += 4) {
//     aux = _mm_set_ps(x[i], x[i], x[i], x[i]);
//     a_rx = _mm_sub_ps(_mm_load_ps(x + j), aux);
//     aux = _mm_set_ps(y[i], y[i], y[i], y[i]);
//     a_ry = _mm_sub_ps(_mm_load_ps(y + j), aux);
//     aux = _mm_set_ps(z[i], z[i], z[i], z[i]);
//     a_rz = _mm_sub_ps(_mm_load_ps(z + j), aux);
//
//     a_r2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(a_rx, a_rx), a_eps),
//                       _mm_add_ps(_mm_mul_ps(a_ry, a_ry),
//                                  _mm_mul_ps(a_rz, a_rz)));
//     a_r2inv = _mm_rsqrt_ps(a_r2);
//     a_r6inv = _mm_mul_ps(a_r2inv, _mm_mul_ps(a_r2inv, a_r2inv));
//     a_s = _mm_mul_ps(_mm_load_ps(m + j), a_r6inv);
//
//     a_rx = _mm_mul_ps(a_s, a_rx);
//     a_rx = _mm_hadd_ps(a_rx, a_rx);
//     a_rx = _mm_hadd_ps(a_rx, a_rx);
//     ax[i] += a_rx[0];
//     a_ry = _mm_mul_ps(a_s, a_ry);
//     a_ry = _mm_hadd_ps(a_ry, a_ry);
//     a_ry = _mm_hadd_ps(a_ry, a_ry);
//     ay[i] += a_ry[0];
//     a_rz = _mm_mul_ps(a_s, a_rz);
//     a_rz = _mm_hadd_ps(a_rz, a_rz);
//     a_rz = _mm_hadd_ps(a_rz, a_rz);
//     az[i] += a_rz[0];
//   }
//
//   for (; j < N; j++) {
//     float rx = x[j] - x[i];
//     float ry = y[j] - y[i];
//     float rz = z[j] - z[i];
//     float r2 = rx*rx + ry*ry + rz*rz + eps;
//     float r2inv = 1.0f / sqrt(r2);
//     float r6inv = r2inv * r2inv * r2inv;
//     float s = m[j] * r6inv;
//     ax[i] += s * rx;
//     ay[i] += s * ry;
//     az[i] += s * rz;
//   }
// }
