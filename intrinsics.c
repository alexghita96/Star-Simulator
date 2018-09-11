#include <immintrin.h>

void compute() {
  double t0, t1;

  int i, j; // for loop counters
  int j1, j2, j3, j4, j5; // counter offsets used in pipeline
  int unroll_N = (N / 4) * 4; // unroll counter; unroll factor is always 4

  // pipeline arrays used in Loop 1
  float* rx = (float*) _mm_malloc(N * sizeof(float), 16);
  float* ry = (float*) _mm_malloc(N * sizeof(float), 16);
  float* rz = (float*) _mm_malloc(N * sizeof(float), 16);
  float* r2 = (float*) _mm_malloc(N * sizeof(float), 16);
  float* r2inv = (float*) _mm_malloc(N * sizeof(float), 16);
  float* r6inv = (float*) _mm_malloc(N * sizeof(float), 16);
  float* s = (float*) _mm_malloc(N * sizeof(float), 16);

  // constants vectors used in computations
  __m128 a_eps = _mm_set_ps(eps, eps, eps, eps);
  __m128 a_dt = _mm_set_ps(dt, dt, dt, dt);
  __m128 a_dmp = _mm_set_ps(dmp, dmp, dmp, dmp);
  __m128 a_zeros = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
  __m128 a_ones = _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f);
  __m128 a_mones = _mm_set_ps(-1.0f, -1.0f, -1.0f, -1.0f);

  // vectors used to replace the variables used in the original code
  __m128 a_rx, a_ry, a_rz, a_r2, a_r2inv, a_r6inv, a_s, a_m;
  __m128 a_x, a_y, a_z, aux;
  __m128 a_a, a_v, a_cmp;
  __m128 a_xi, a_yi, a_zi;

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

  // ax loops
  for (i = 0; i < unroll_N; i += 4) {
    _mm_store_ps(ax + i, a_zeros);
  }
  for (; i < N; i++) {
    ax[i] = 0.0f;
  }

  // ay loops
  for (i = 0; i < unroll_N; i += 4) {
    _mm_store_ps(ay + i, a_zeros);
  }
  for (; i < N; i++) {
    ay[i] = 0.0f;
  }

  // az loops
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
    * 1. The loop is pipelined, as several distinct computational steps have
    *    been identified in the inner loop, each of them depending on a
    *    previous one, making this loop an ideal candidate for pipelining. A
    *    prolog and epilog are added, and the computed values are stored in
    *    arrays, so that values from up to five iterations before can be used.
    * 2. The prolog, pipeline and epilog are unrolled, and an ending,
    *    non-pipelined loop is added to finish the remaining computations.
    * 3. The resulting loop is vectorised for an additional speedup.
    */

  t0 = wtime();

  for (i = 0; i < N; i++) {
    // load the i values once, as they are used throughout the loops below
    a_xi = _mm_load_ps1(x + i);
    a_yi = _mm_load_ps1(y + i);
    a_zi = _mm_load_ps1(z + i);

    // prolog
    for (j = 0; j < 20; j += 4) {
      // rx, ry, rz computations
      _mm_store_ps(rx + j, _mm_sub_ps(_mm_load_ps(x + j), a_xi));
      _mm_store_ps(ry + j, _mm_sub_ps(_mm_load_ps(y + j), a_yi));
      _mm_store_ps(rz + j, _mm_sub_ps(_mm_load_ps(z + j), a_zi));

      // second iteration and beyond
      if (j > 0) {
        j1 = j - 4;

        // r2 computation
        a_rx = _mm_load_ps(rx + j1);
        a_ry = _mm_load_ps(ry + j1);
        a_rz = _mm_load_ps(rz + j1);
        _mm_store_ps(r2 + j1, _mm_add_ps(_mm_add_ps(_mm_mul_ps(a_rx, a_rx),
                                                    _mm_mul_ps(a_ry, a_ry)),
                                         _mm_add_ps(_mm_mul_ps(a_rz, a_rz),
                                                    a_eps)));

        // third iteration and beyond
        if (j > 4) {
          j2 = j - 8;

          // r2inv computation
          a_r2 = _mm_load_ps(r2 + j2);
          _mm_store_ps(r2inv + j2, _mm_rsqrt_ps(a_r2));

          // fourth iteration and beyond
          if (j > 8) {
            j3 = j - 12;

            // r6inv computation
            a_r2inv = _mm_load_ps(r2inv + j3);
            _mm_store_ps(r6inv + j3, _mm_mul_ps(a_r2inv,
                                                _mm_mul_ps(a_r2inv,
                                                           a_r2inv)));

            // fifth iteration
            if (j > 12) {
              j4 = j - 16;

              // s computation
              a_m = _mm_load_ps(m + j4);
              a_r6inv = _mm_load_ps(r6inv + j4);
              _mm_store_ps(s + j4, _mm_mul_ps(a_m, a_r6inv));
            }
          }
        }
      }
    }

    // pipeline
    for (; j < unroll_N; j += 4) {
      j1 = j - 4;
      j2 = j - 8;
      j3 = j - 12;
      j4 = j - 16;
      j5 = j - 20;

      // pipeline step 1: rx, ry, rz computations
      _mm_store_ps(rx + j, _mm_sub_ps(_mm_load_ps(x + j), a_xi));
      _mm_store_ps(ry + j, _mm_sub_ps(_mm_load_ps(y + j), a_yi));
      _mm_store_ps(rz + j, _mm_sub_ps(_mm_load_ps(z + j), a_zi));

      // pipeline step 2: r2 computation
      a_rx = _mm_load_ps(rx + j1);
      a_ry = _mm_load_ps(ry + j1);
      a_rz = _mm_load_ps(rz + j1);
      _mm_store_ps(r2 + j1, _mm_add_ps(_mm_add_ps(_mm_mul_ps(a_rx, a_rx),
                                                  _mm_mul_ps(a_ry, a_ry)),
                                       _mm_add_ps(_mm_mul_ps(a_rz, a_rz),
                                                  a_eps)));

      // pipeline step 3: r2inv computation
      a_r2 = _mm_load_ps(r2 + j2);
      _mm_store_ps(r2inv + j2, _mm_rsqrt_ps(a_r2));

      // pipeline step 4: r6inv computation
      a_r2inv = _mm_load_ps(r2inv + j3);
      _mm_store_ps(r6inv + j3, _mm_mul_ps(a_r2inv, _mm_mul_ps(a_r2inv,
                                                              a_r2inv)));

      // pipeline step 5: s computation
      a_m = _mm_load_ps(m + j4);
      a_r6inv = _mm_load_ps(r6inv + j4);
      _mm_store_ps(s + j4, _mm_mul_ps(a_m, a_r6inv));

      // pipeline step 6: ax, ay, az computations
      a_s = _mm_load_ps(s + j5);
      a_rx = _mm_load_ps(rx + j5);
      a_rx = _mm_mul_ps(a_s, a_rx);
      a_rx = _mm_hadd_ps(a_rx, a_rx);
      a_rx = _mm_hadd_ps(a_rx, a_rx);
      ax[i] += a_rx[0];

      a_ry = _mm_load_ps(ry + j5);
      a_ry = _mm_mul_ps(a_s, a_ry);
      a_ry = _mm_hadd_ps(a_ry, a_ry);
      a_ry = _mm_hadd_ps(a_ry, a_ry);
      ay[i] += a_ry[0];

      a_rz = _mm_load_ps(rz + j5);
      a_rz = _mm_mul_ps(a_s, a_rz);
      a_rz = _mm_hadd_ps(a_rz, a_rz);
      a_rz = _mm_hadd_ps(a_rz, a_rz);
      az[i] += a_rz[0];
    }

    // epilog
    for (j = unroll_N - 20; j < unroll_N; j += 4) {
      // first iteration
      if (j < unroll_N - 16) {
        j1 = j + 16;

        // r2 computation
        a_rx = _mm_load_ps(rx + j1);
        a_ry = _mm_load_ps(ry + j1);
        a_rz = _mm_load_ps(rz + j1);
        _mm_store_ps(r2 + j1, _mm_add_ps(_mm_add_ps(_mm_mul_ps(a_rx, a_rx),
                                                    _mm_mul_ps(a_ry, a_ry)),
                                         _mm_add_ps(_mm_mul_ps(a_rz, a_rz),
                                                    a_eps)));
      }

      // up to the second iteration
      if (j < unroll_N - 12) {
        j2 = j + 12;

        // r2inv computation
        a_r2 = _mm_load_ps(r2 + j2);
        _mm_store_ps(r2inv + j2, _mm_rsqrt_ps(a_r2));
      }

      // up to the third iteration
      if (j < unroll_N - 8) {
        j3 = j + 8;

        // r6inv computation
        a_r2inv = _mm_load_ps(r2inv + j3);
        _mm_store_ps(r6inv + j3, _mm_mul_ps(a_r2inv,
                                            _mm_mul_ps(a_r2inv,
                                                       a_r2inv)));
      }

      // up to the fourth iteration
      if (j < unroll_N - 4) {
        j4 = j + 4;

        // s computation
        a_m = _mm_load_ps(m + j4);
        a_r6inv = _mm_load_ps(r6inv + j4);
        _mm_store_ps(s + j4, _mm_mul_ps(a_m, a_r6inv));
      }

      // ax, ay, az computations
      a_s = _mm_load_ps(s + j);
      a_rx = _mm_load_ps(rx + j);
      a_rx = _mm_mul_ps(a_s, a_rx);
      a_rx = _mm_hadd_ps(a_rx, a_rx);
      a_rx = _mm_hadd_ps(a_rx, a_rx);
      ax[i] += a_rx[0];

      a_ry = _mm_load_ps(ry + j);
      a_ry = _mm_mul_ps(a_s, a_ry);
      a_ry = _mm_hadd_ps(a_ry, a_ry);
      a_ry = _mm_hadd_ps(a_ry, a_ry);
      ay[i] += a_ry[0];

      a_rz = _mm_load_ps(rz + j);
      a_rz = _mm_mul_ps(a_s, a_rz);
      a_rz = _mm_hadd_ps(a_rz, a_rz);
      a_rz = _mm_hadd_ps(a_rz, a_rz);
      az[i] += a_rz[0];
    }

    // unroll tidy-up
    for (; j < N; j++) {
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

  // free the arrays used in Loop 1, as they are no longer needed
  _mm_free(rx);
  _mm_free(ry);
  _mm_free(rz);
  _mm_free(r2);
  _mm_free(r2inv);
  _mm_free(r6inv);
  _mm_free(s);

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

  t0 = wtime();

  // x loops
  for (i = 0; i < unroll_N; i += 4) {
    // vx[i] += dmp * (dt * ax[i]);
    a_v = _mm_load_ps(vx + i);
    a_a = _mm_load_ps(ax + i);
    a_v = _mm_add_ps(a_v, _mm_mul_ps(a_dmp, _mm_mul_ps(a_dt, a_a)));

    // x[i] += dt * vx[i];
    a_x = _mm_load_ps(x + i);
    a_x = _mm_add_ps(a_x, _mm_mul_ps(a_dt, a_v));

    // if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
    a_cmp = _mm_or_ps(_mm_cmpge_ps(a_x, a_ones), _mm_cmple_ps(a_x, a_mones));
    a_cmp = _mm_or_ps(a_cmp, a_ones);
    a_cmp = _mm_and_ps(a_cmp, a_mones);
    a_v = _mm_mul_ps(a_v, a_cmp);

    // storage
    _mm_store_ps(x + i, a_x);
    _mm_store_ps(vx + i, a_v);
  }
  for (; i < unroll_N; i++) {
    vx[i] += dmp * (dt * ax[i]);
    x[i] += dt * vx[i];
    if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
  }

  // y loops
  for (i = 0; i < unroll_N; i += 4) {
    // vy[i] += dmp * (dt * ay[i]);
    a_v = _mm_load_ps(vy + i);
    a_a = _mm_load_ps(ay + i);
    a_v = _mm_add_ps(a_v, _mm_mul_ps(a_dmp, _mm_mul_ps(a_dt, a_a)));

    // y[i] += dt * vy[i];
    a_y = _mm_load_ps(y + i);
    a_y = _mm_add_ps(a_y, _mm_mul_ps(a_dt, a_v));

    // if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
    a_cmp = _mm_or_ps(_mm_cmpge_ps(a_y, a_ones), _mm_cmple_ps(a_y, a_mones));
    a_cmp = _mm_or_ps(a_cmp, a_ones);
    a_cmp = _mm_and_ps(a_cmp, a_mones);
    a_v = _mm_mul_ps(a_v, a_cmp);

    // storage
    _mm_store_ps(y + i, a_y);
    _mm_store_ps(vy + i, a_v);
  }
  for (; i < unroll_N; i++) {
    vy[i] += dmp * (dt * ay[i]);
    y[i] += dt * vy[i];
    if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
  }

  // z loops
  for (i = 0; i < unroll_N; i += 4) {
    // vz[i] += dmp * (dt * az[i]);
    a_v = _mm_load_ps(vz + i);
    a_a = _mm_load_ps(az + i);
    a_v = _mm_add_ps(a_v, _mm_mul_ps(a_dmp, _mm_mul_ps(a_dt, a_a)));

    // z[i] += dt * vz[i];
    a_z = _mm_load_ps(z + i);
    a_z = _mm_add_ps(a_z, _mm_mul_ps(a_dt, a_v));

    // if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;
    a_cmp = _mm_or_ps(_mm_cmpge_ps(a_z, a_ones), _mm_cmple_ps(a_z, a_mones));
    a_cmp = _mm_or_ps(a_cmp, a_ones);
    a_cmp = _mm_and_ps(a_cmp, a_mones);
    a_v = _mm_mul_ps(a_v, a_cmp);

    // storage
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
