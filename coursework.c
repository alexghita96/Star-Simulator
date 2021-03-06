#include <immintrin.h>
#include <omp.h>

float hsum_ps_sse3(__m128 v) {
  __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
  __m128 sums = _mm_add_ps(v, shuf);
  shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
  sums        = _mm_add_ss(sums, shuf);
  return        _mm_cvtss_f32(sums);
}

void compute() {
  omp_set_num_threads(8);

  double t0, t1;

  int i, j; // For loop counters.
  int unroll_N = (N / 4) * 4; // Unroll counter; unroll factor is always 4.

  // Constants vectors used in computations.
  __m128 a_eps = _mm_set_ps(eps, eps, eps, eps);
  __m128 a_zeros = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);

  /****************************************************************************
   *                                  Loop 0                                  *
   ****************************************************************************/

  /**
   * Optimisation steps:
   * 1. The loop is fissioned, in order to make use of spatial locality by
   *    focussing on one array at a time.
   * 2. Each of the three loops is unrolled, so an ending loop is added for
   *    each of them to compute the remaining N % 4 values.
   * 3. The three loops (excluding those added in step 2) are vectorised for an
   *    additional speedup. a_zeros, a vector filled with 0.0f is used to set
   *    the three arrays' values in chunks of four.
   */

  t0 = wtime();

  // ax loops.
  for (i = 0; i < unroll_N; i += 4) {
    _mm_store_ps(ax + i, a_zeros);
  }
  for (; i < N; i++) {
    ax[i] = 0.0f;
  }

  // ay loops.
  for (i = 0; i < unroll_N; i += 4) {
    _mm_store_ps(ay + i, a_zeros);
  }
  for (; i < N; i++) {
    ay[i] = 0.0f;
  }

  // az loops.
  for (i = 0; i < unroll_N; i += 4) {
    _mm_store_ps(az + i, a_zeros);
  }
  for (; i < N; i++) {
    az[i] = 0.0f;
  }

  t1 = wtime();
  l0 += (t1 - t0);

  /****************************************************************************
   *                                  Loop 1                                  *
   ****************************************************************************/

  /**
   * Optimisation steps:
   * 1. The inner loop is unrolled, so an ending loop is added to compute the
   *    remaining iterations.
   * 2. The inner loop is vectorised. As many vectors are declared inside the
   *    loop as possible, so that false sharing is avoided in the next step.
   * 3. Both loops are then parallelised for an additional speedup.
   */

  t0 = wtime();

  // This type of scheduling was found to work best.
  #pragma omp parallel for schedule(guided)
  for (i = 0; i < N; i++) {
    // The results of the inner loop iterations will be stored here.
    double tx = 0.0f, ty = 0.0f, tz = 0.0f;

    // Initiliase i-dependent values outside the inner loop.
    __m128 aux1 = _mm_set_ps(x[i], x[i], x[i], x[i]);
    __m128 aux2 = _mm_set_ps(y[i], y[i], y[i], y[i]);
    __m128 aux3 = _mm_set_ps(z[i], z[i], z[i], z[i]);

    #pragma omp parallel for schedule(guided) reduction(+:tx, ty, tz) firstprivate(aux1, aux2, aux3)
    for (j = 0; j < unroll_N; j += 4) {
      // For each variable in the origin loop, use an a_variable vector.
      __m128 a_rx, a_ry, a_rz, a_r2, a_r2inv, a_r6inv, a_s;

      a_rx = _mm_sub_ps(_mm_load_ps(x + j), aux1);
      a_ry = _mm_sub_ps(_mm_load_ps(y + j), aux2);
      a_rz = _mm_sub_ps(_mm_load_ps(z + j), aux3);

      a_r2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(a_rx, a_rx), a_eps),
                        _mm_add_ps(_mm_mul_ps(a_ry, a_ry),
                                   _mm_mul_ps(a_rz, a_rz)));

      a_r2inv = _mm_rsqrt_ps(a_r2);

      a_r6inv = _mm_mul_ps(a_r2inv, _mm_mul_ps(a_r2inv, a_r2inv));
      a_s = _mm_mul_ps(_mm_load_ps(m + j), a_r6inv);

      // Store the results in the shared t-variables.
      a_rx = _mm_mul_ps(a_s, a_rx);
      tx += hsum_ps_sse3(a_rx);
      a_ry = _mm_mul_ps(a_s, a_ry);
      ty += hsum_ps_sse3(a_ry);
      a_rz = _mm_mul_ps(a_s, a_rz);
      tz += hsum_ps_sse3(a_rz);
    }

    // Update the arrays with the stored variables.
    ax[i] += tx;
    ay[i] += ty;
    az[i] += tz;

    // Tidy-up loop for unroll. No need to parallelise, as it has few steps.
    for (j = unroll_N; j < N; j++) {
      float rx = x[j] - x[i];
      float ry = y[j] - y[i];
      float rz = z[j] - z[i];
      float r2 = rx*rx + ry*ry + rz*rz + eps;
      float r2inv = 1.0f / sqrt(r2);
      float r6inv = r2inv * r2inv * r2inv;
      float s = m[j] * r6inv;
      ax[i] += s * rx;
      ay[i] += s * ry;
      az[i] += s * rz;
    }
  }

  t1 = wtime();
  l1 += (t1 - t0);

  /****************************************************************************
   *                              Loop 2 & Loop 3                             *
   ****************************************************************************/

  /**
   * Optimisation steps:
   * 1. The loop is fissioned, in order to make use of spatial locality by
   *    focussing on one array at a time.
   * 2. Loop 3 is also fissioned for the same reasons.
   * 3. Corresponding loops (x loops, y loops and z loops) from Loop 2 and
   *    Loop 3 are fusioned, as they refer to the same arrays and this way
   *    spatial locality will be exploited.
   * 4. Each loop is unrolled and an ending loop is added to each of them to
   *    perform the remaining computations.
   * 5. Each loop is vectorised for an additional speedup.
   */

   __m128 a_dt = _mm_set_ps(dt, dt, dt, dt);
   __m128 a_dmp = _mm_set_ps(dmp, dmp, dmp, dmp);
   __m128 a_ones = _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f);
   __m128 a_mones = _mm_set_ps(-1.0f, -1.0f, -1.0f, -1.0f);

   // Vectors used to replace the variables used in the original code.
   __m128 a_x, a_y, a_z;
   __m128 a_a, a_v, a_cmp;

  t0 = wtime();

  // x loops.
  for (i = 0; i < unroll_N; i += 4) {
    // Equivalent to: vx[i] += dmp * (dt * ax[i]).
    a_v = _mm_load_ps(vx + i);
    a_a = _mm_load_ps(ax + i);
    a_v = _mm_add_ps(a_v, _mm_mul_ps(a_dmp, _mm_mul_ps(a_dt, a_a)));

    // Equivalent to: x[i] += dt * vx[i].
    a_x = _mm_load_ps(x + i);
    a_x = _mm_add_ps(a_x, _mm_mul_ps(a_dt, a_v));

    // Equivalent to: if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f.
    a_cmp = _mm_or_ps(_mm_cmpge_ps(a_x, a_ones), _mm_cmple_ps(a_x, a_mones));
    a_cmp = _mm_or_ps(a_cmp, a_ones);
    a_cmp = _mm_and_ps(a_cmp, a_mones);
    a_v = _mm_mul_ps(a_v, a_cmp);

    // Storage.
    _mm_store_ps(x + i, a_x);
    _mm_store_ps(vx + i, a_v);
  }
  for (; i < unroll_N; i++) {
    vx[i] += dmp * (dt * ax[i]);
    x[i] += dt * vx[i];
    if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
  }

  // y loops.
  for (i = 0; i < unroll_N; i += 4) {
    // Equivalent to: vy[i] += dmp * (dt * ay[i]).
    a_v = _mm_load_ps(vy + i);
    a_a = _mm_load_ps(ay + i);
    a_v = _mm_add_ps(a_v, _mm_mul_ps(a_dmp, _mm_mul_ps(a_dt, a_a)));

    // Equivalent to: y[i] += dt * vy[i].
    a_y = _mm_load_ps(y + i);
    a_y = _mm_add_ps(a_y, _mm_mul_ps(a_dt, a_v));

    // Equivalent to: if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f.
    a_cmp = _mm_or_ps(_mm_cmpge_ps(a_y, a_ones), _mm_cmple_ps(a_y, a_mones));
    a_cmp = _mm_or_ps(a_cmp, a_ones);
    a_cmp = _mm_and_ps(a_cmp, a_mones);
    a_v = _mm_mul_ps(a_v, a_cmp);

    // Storage.
    _mm_store_ps(y + i, a_y);
    _mm_store_ps(vy + i, a_v);
  }
  for (; i < unroll_N; i++) {
    vy[i] += dmp * (dt * ay[i]);
    y[i] += dt * vy[i];
    if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
  }

  // z loops.
  for (i = 0; i < unroll_N; i += 4) {
    // Equivalent to: vz[i] += dmp * (dt * az[i]).
    a_v = _mm_load_ps(vz + i);
    a_a = _mm_load_ps(az + i);
    a_v = _mm_add_ps(a_v, _mm_mul_ps(a_dmp, _mm_mul_ps(a_dt, a_a)));

    // Equivalent to: z[i] += dt * vz[i].
    a_z = _mm_load_ps(z + i);
    a_z = _mm_add_ps(a_z, _mm_mul_ps(a_dt, a_v));

    // Equivalent to: if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f.
    a_cmp = _mm_or_ps(_mm_cmpge_ps(a_z, a_ones), _mm_cmple_ps(a_z, a_mones));
    a_cmp = _mm_or_ps(a_cmp, a_ones);
    a_cmp = _mm_and_ps(a_cmp, a_mones);
    a_v = _mm_mul_ps(a_v, a_cmp);

    // Storage.
    _mm_store_ps(z + i, a_z);
    _mm_store_ps(vz + i, a_v);
  }
  for (; i < unroll_N; i++) {
    vz[i] += dmp * (dt * az[i]);
    z[i] += dt * vz[i];
    if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;
  }

  t1 = wtime();
  l2 += (t1 - t0);
}

/******************************************************************************
 * Below is a pipelined version of Loop 1 that was used in the serial version *
 * of the program, for comparison purposes.                                   *
 ******************************************************************************
 ******************************************************************************
 *
 *
 * // pipeline arrays used in Loop 1
 * float* rx = (float*) _mm_malloc(N * sizeof(float), 16);
 * float* ry = (float*) _mm_malloc(N * sizeof(float), 16);
 * float* rz = (float*) _mm_malloc(N * sizeof(float), 16);
 * float* r2 = (float*) _mm_malloc(N * sizeof(float), 16);
 * float* r2inv = (float*) _mm_malloc(N * sizeof(float), 16);
 * float* r6inv = (float*) _mm_malloc(N * sizeof(float), 16);
 * float* s = (float*) _mm_malloc(N * sizeof(float), 16);
 *
 * for (i = 0; i < N; i++) {
 *   // load the i values once, as they are used throughout the loops below
 *   a_xi = _mm_load_ps1(x + i);
 *   a_yi = _mm_load_ps1(y + i);
 *   a_zi = _mm_load_ps1(z + i);
 *
 *   // prolog
 *   for (j = 0; j < 20; j += 4) {
 *     // rx, ry, rz computations
 *     _mm_store_ps(rx + j, _mm_sub_ps(_mm_load_ps(x + j), a_xi));
 *     _mm_store_ps(ry + j, _mm_sub_ps(_mm_load_ps(y + j), a_yi));
 *     _mm_store_ps(rz + j, _mm_sub_ps(_mm_load_ps(z + j), a_zi));
 *
 *     // second iteration and beyond
 *     if (j > 0) {
 *       j1 = j - 4;
 *
 *       // r2 computation
 *       a_rx = _mm_load_ps(rx + j1);
 *       a_ry = _mm_load_ps(ry + j1);
 *       a_rz = _mm_load_ps(rz + j1);
 *       _mm_store_ps(r2 + j1, _mm_add_ps(_mm_add_ps(_mm_mul_ps(a_rx, a_rx),
 *                                                   _mm_mul_ps(a_ry, a_ry)),
 *                                        _mm_add_ps(_mm_mul_ps(a_rz, a_rz),
 *                                                   a_eps)));
 *
 *       // third iteration and beyond
 *       if (j > 4) {
 *         j2 = j - 8;
 *
 *         // r2inv computation
 *         a_r2 = _mm_load_ps(r2 + j2);
 *         _mm_store_ps(r2inv + j2, _mm_rsqrt_ps(a_r2));
 *
 *         // fourth iteration and beyond
 *         if (j > 8) {
 *           j3 = j - 12;
 *
 *           // r6inv computation
 *           a_r2inv = _mm_load_ps(r2inv + j3);
 *           _mm_store_ps(r6inv + j3, _mm_mul_ps(a_r2inv,
 *                                               _mm_mul_ps(a_r2inv,
 *                                                          a_r2inv)));
 *
 *           // fifth iteration
 *           if (j > 12) {
 *             j4 = j - 16;
 *
 *             // s computation
 *             a_m = _mm_load_ps(m + j4);
 *             a_r6inv = _mm_load_ps(r6inv + j4);
 *             _mm_store_ps(s + j4, _mm_mul_ps(a_m, a_r6inv));
 *           }
 *         }
 *       }
 *     }
 *   }
 *
 *   // pipeline
 *   for (; j < unroll_N; j += 4) {
 *     j1 = j - 4;
 *     j2 = j - 8;
 *     j3 = j - 12;
 *     j4 = j - 16;
 *     j5 = j - 20;
 *
 *     // pipeline step 1: rx, ry, rz computations
 *     _mm_store_ps(rx + j, _mm_sub_ps(_mm_load_ps(x + j), a_xi));
 *     _mm_store_ps(ry + j, _mm_sub_ps(_mm_load_ps(y + j), a_yi));
 *     _mm_store_ps(rz + j, _mm_sub_ps(_mm_load_ps(z + j), a_zi));
 *
 *     // pipeline step 2: r2 computation
 *     a_rx = _mm_load_ps(rx + j1);
 *     a_ry = _mm_load_ps(ry + j1);
 *     a_rz = _mm_load_ps(rz + j1);
 *     _mm_store_ps(r2 + j1, _mm_add_ps(_mm_add_ps(_mm_mul_ps(a_rx, a_rx),
 *                                                 _mm_mul_ps(a_ry, a_ry)),
 *                                      _mm_add_ps(_mm_mul_ps(a_rz, a_rz),
 *                                                 a_eps)));
 *
 *     // pipeline step 3: r2inv computation
 *     a_r2 = _mm_load_ps(r2 + j2);
 *     _mm_store_ps(r2inv + j2, _mm_rsqrt_ps(a_r2));
 *
 *     // pipeline step 4: r6inv computation
 *     a_r2inv = _mm_load_ps(r2inv + j3);
 *     _mm_store_ps(r6inv + j3, _mm_mul_ps(a_r2inv, _mm_mul_ps(a_r2inv,
 *                                                             a_r2inv)));
 *
 *     // pipeline step 5: s computation
 *     a_m = _mm_load_ps(m + j4);
 *     a_r6inv = _mm_load_ps(r6inv + j4);
 *     _mm_store_ps(s + j4, _mm_mul_ps(a_m, a_r6inv));
 *
 *     // pipeline step 6: ax, ay, az computations
 *     a_s = _mm_load_ps(s + j5);
 *     a_rx = _mm_load_ps(rx + j5);
 *     a_rx = _mm_mul_ps(a_s, a_rx);
 *     a_rx = _mm_hadd_ps(a_rx, a_rx);
 *     a_rx = _mm_hadd_ps(a_rx, a_rx);
 *     ax[i] += a_rx[0];
 *
 *     a_ry = _mm_load_ps(ry + j5);
 *     a_ry = _mm_mul_ps(a_s, a_ry);
 *     a_ry = _mm_hadd_ps(a_ry, a_ry);
 *     a_ry = _mm_hadd_ps(a_ry, a_ry);
 *     ay[i] += a_ry[0];
 *
 *     a_rz = _mm_load_ps(rz + j5);
 *     a_rz = _mm_mul_ps(a_s, a_rz);
 *     a_rz = _mm_hadd_ps(a_rz, a_rz);
 *     a_rz = _mm_hadd_ps(a_rz, a_rz);
 *     az[i] += a_rz[0];
 *   }
 *
 *   // epilog
 *   for (j = unroll_N - 20; j < unroll_N; j += 4) {
 *     // first iteration
 *     if (j < unroll_N - 16) {
 *       j1 = j + 16;
 *
 *       // r2 computation
 *       a_rx = _mm_load_ps(rx + j1);
 *       a_ry = _mm_load_ps(ry + j1);
 *       a_rz = _mm_load_ps(rz + j1);
 *       _mm_store_ps(r2 + j1, _mm_add_ps(_mm_add_ps(_mm_mul_ps(a_rx, a_rx),
 *                                                   _mm_mul_ps(a_ry, a_ry)),
 *                                        _mm_add_ps(_mm_mul_ps(a_rz, a_rz),
 *                                                    a_eps)));
 *     }
 *
 *     // up to the second iteration
 *     if (j < unroll_N - 12) {
 *       j2 = j + 12;
 *
 *       // r2inv computation
 *       a_r2 = _mm_load_ps(r2 + j2);
 *       _mm_store_ps(r2inv + j2, _mm_rsqrt_ps(a_r2));
 *     }
 *
 *     // up to the third iteration
 *     if (j < unroll_N - 8) {
 *       j3 = j + 8;
 *
 *       // r6inv computation
 *       a_r2inv = _mm_load_ps(r2inv + j3);
 *       _mm_store_ps(r6inv + j3, _mm_mul_ps(a_r2inv,
 *                                           _mm_mul_ps(a_r2inv,
 *                                                      a_r2inv)));
 *     }
 *
 *     // up to the fourth iteration
 *     if (j < unroll_N - 4) {
 *       j4 = j + 4;
 *
 *       // s computation
 *       a_m = _mm_load_ps(m + j4);
 *       a_r6inv = _mm_load_ps(r6inv + j4);
 *       _mm_store_ps(s + j4, _mm_mul_ps(a_m, a_r6inv));
 *     }
 *
 *     // ax, ay, az computations
 *     a_s = _mm_load_ps(s + j);
 *     a_rx = _mm_load_ps(rx + j);
 *     a_rx = _mm_mul_ps(a_s, a_rx);
 *     a_rx = _mm_hadd_ps(a_rx, a_rx);
 *     a_rx = _mm_hadd_ps(a_rx, a_rx);
 *     ax[i] += a_rx[0];
 *
 *     a_ry = _mm_load_ps(ry + j);
 *     a_ry = _mm_mul_ps(a_s, a_ry);
 *     a_ry = _mm_hadd_ps(a_ry, a_ry);
 *     a_ry = _mm_hadd_ps(a_ry, a_ry);
 *     ay[i] += a_ry[0];
 *
 *     a_rz = _mm_load_ps(rz + j);
 *     a_rz = _mm_mul_ps(a_s, a_rz);
 *     a_rz = _mm_hadd_ps(a_rz, a_rz);
 *     a_rz = _mm_hadd_ps(a_rz, a_rz);
 *     az[i] += a_rz[0];
 *   }
 *
 *   // unroll tidy-up
 *   for (; j < N; j++) {
 *     float rx = x[j] - x[i];
 *     float ry = y[j] - y[i];
 *     float rz = z[j] - z[i];
 *     float r2 = rx*rx + ry*ry + rz*rz + eps;
 *     float r2inv = 1.0f / sqrt(r2);
 *     float r6inv = r2inv * r2inv * r2inv;
 *     float s = m[j] * r6inv;
 *     ax[i] += s * rx;
 *     ay[i] += s * ry;
 *     az[i] += s * rz;
 *   }
 * }
 *
 * // free the arrays used in Loop 1, as they are no longer needed
 * _mm_free(rx);
 * _mm_free(ry);
 * _mm_free(rz);
 * _mm_free(r2);
 * _mm_free(r2inv);
 * _mm_free(r6inv);
 * _mm_free(s);
 */
