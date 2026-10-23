# Dataset audit

## Parsing rules
Files are parsed as `<network>_<leak code>_<flow>[_<noise>]_sensor.ext` under a sensor/network/leak folder. Codes are LO/BR, LC/CC/GL/OL/NL, flows ND/0.18 LPS/0.47 LPS/Transient, and channels A1/A2, P1/P2, H1/H2. No filename or folder token contains leak position, sensor position, leak size, or leak flow.

## Verified inventory

sensor_type
Accelerometer               80
Dynamic Pressure Sensor     80
Hydrophone                 122

sensor_type              parse_ok
Accelerometer            True         80
Dynamic Pressure Sensor  True         80
Hydrophone               False         2
                         True        120

### Sample metadata

| path                                                                | sensor_type   | folder                |   size_bytes | parse_ok   | network   | leak_code   | leak_type             | flow      | noise   | channel   | signal_id         | columns      |   samples |   fs_hz |   duration_s |   nan_count | constant   |   duplicate_count |         min |        max |         mean |        std |   outlier_count_robust |
|:--------------------------------------------------------------------|:--------------|:----------------------|-------------:|:-----------|:----------|:------------|:----------------------|:----------|:--------|:----------|:------------------|:-------------|----------:|--------:|-------------:|------------:|:-----------|------------------:|------------:|-----------:|-------------:|-----------:|-----------------------:|
| Accelerometer\Branched\Circumferential Crack\BR_CC_0.18 LPS_A1.csv  | Accelerometer | Circumferential Crack |     23109495 | True       | BR        | CC          | Circumferential Crack | 0.18 LPS  | N       | A1        | BR_CC_0.18 LPS_N  | Sample|Value |    923488 | 25601.6 |      36.0714 |           0 | False      |            867695 | -0.0309617  | 0.0291752  |  0.000139715 | 0.00575969 |                      0 |
| Accelerometer\Branched\Circumferential Crack\BR_CC_0.18 LPS_A2.csv  | Accelerometer | Circumferential Crack |     22776017 | True       | BR        | CC          | Circumferential Crack | 0.18 LPS  | N       | A2        | BR_CC_0.18 LPS_N  | Sample|Value |    917996 | 25601.6 |      35.8569 |           0 | False      |            904467 | -0.0051483  | 0.0052687  |  0.000202138 | 0.00118706 |                      0 |
| Accelerometer\Branched\Circumferential Crack\BR_CC_0.47 LPS_A1.csv  | Accelerometer | Circumferential Crack |     22481098 | True       | BR        | CC          | Circumferential Crack | 0.47 LPS  | N       | A1        | BR_CC_0.47 LPS_N  | Sample|Value |    896801 | 25601.6 |      35.029  |           0 | False      |            782945 | -0.0679484  | 0.0691201  | -0.000498775 | 0.0133598  |                      0 |
| Accelerometer\Branched\Circumferential Crack\BR_CC_0.47 LPS_A2.csv  | Accelerometer | Circumferential Crack |     23174652 | True       | BR        | CC          | Circumferential Crack | 0.47 LPS  | N       | A2        | BR_CC_0.47 LPS_N  | Sample|Value |    918966 | 25601.6 |      35.8948 |           0 | False      |            893677 | -0.0134603  | 0.00615915 | -0.00150396  | 0.00222344 |                      0 |
| Accelerometer\Branched\Circumferential Crack\BR_CC_ND_A1.csv        | Accelerometer | Circumferential Crack |     23150957 | True       | BR        | CC          | Circumferential Crack | ND        | N       | A1        | BR_CC_ND_N        | Sample|Value |    933603 | 25601.6 |      36.4665 |           0 | False      |            921992 | -0.00430788 | 0.00509654 |  0.000147143 | 0.00103413 |                      0 |
| Accelerometer\Branched\Circumferential Crack\BR_CC_ND_A2.csv        | Accelerometer | Circumferential Crack |     23406829 | True       | BR        | CC          | Circumferential Crack | ND        | N       | A2        | BR_CC_ND_N        | Sample|Value |    940956 | 25601.6 |      36.7537 |           0 | False      |            923735 | -0.00670228 | 0.00685054 |  0.000215912 | 0.00157667 |                      0 |
| Accelerometer\Branched\Circumferential Crack\BR_CC_Transient_A1.csv | Accelerometer | Circumferential Crack |     23183850 | True       | BR        | CC          | Circumferential Crack | Transient | N       | A1        | BR_CC_Transient_N | Sample|Value |    929361 | 25601.6 |      36.3008 |           0 | False      |            817186 | -0.136264   | 0.140526   |  0.000142669 | 0.011848   |                    312 |
| Accelerometer\Branched\Circumferential Crack\BR_CC_Transient_A2.csv | Accelerometer | Circumferential Crack |     23298066 | True       | BR        | CC          | Circumferential Crack | Transient | N       | A2        | BR_CC_Transient_N | Sample|Value |    936567 | 25601.6 |      36.5823 |           0 | False      |            913683 | -0.216215   | 0.238182   |  0.000203801 | 0.00186568 |                     53 |
| Accelerometer\Branched\Gasket Leak\BR_GL_0.18 LPS_A1.csv            | Accelerometer | Gasket Leak           |     23070527 | True       | BR        | GL          | Gasket Leak           | 0.18 LPS  | N       | A1        | BR_GL_0.18 LPS_N  | Sample|Value |    920697 | 25601.6 |      35.9624 |           0 | False      |            862911 | -0.02819    | 0.0272346  | -0.000400623 | 0.00598346 |                      0 |
| Accelerometer\Branched\Gasket Leak\BR_GL_0.18 LPS_A2.csv            | Accelerometer | Gasket Leak           |     23143519 | True       | BR        | GL          | Gasket Leak           | 0.18 LPS  | N       | A2        | BR_GL_0.18 LPS_N  | Sample|Value |    920954 | 25601.6 |      35.9725 |           0 | False      |            888657 | -0.0152383  | 0.0119243  | -0.00105105  | 0.00293757 |                      0 |
| Accelerometer\Branched\Gasket Leak\BR_GL_0.47 LPS_A1.csv            | Accelerometer | Gasket Leak           |     23155579 | True       | BR        | GL          | Gasket Leak           | 0.47 LPS  | N       | A1        | BR_GL_0.47 LPS_N  | Sample|Value |    924328 | 25601.6 |      36.1043 |           0 | False      |            804565 | -0.0620465  | 0.0618116  |  0.000194516 | 0.0141027  |                      0 |
| Accelerometer\Branched\Gasket Leak\BR_GL_0.47 LPS_A2.csv            | Accelerometer | Gasket Leak           |     23366847 | True       | BR        | GL          | Gasket Leak           | 0.47 LPS  | N       | A2        | BR_GL_0.47 LPS_N  | Sample|Value |    935084 | 25601.6 |      36.5244 |           0 | False      |            896458 | -0.0155392  | 0.0160583  |  0.000151336 | 0.0036196  |                      0 |

### Class/condition counts

leak_type              flow       noise  network
Circumferential Crack  0.18 LPS   N      BR         6
                                         LO         6
                       0.47 LPS   N      BR         6
                                         LO         6
                       ND         N      BR         6
                                         LO         6
                                  NN     BR         2
                                         LO         2
                       Transient  N      BR         6
                                         LO         6
                                  NN     BR         2
                                         LO         2
Gasket Leak            0.18 LPS   N      BR         6
                                         LO         6
                       0.47 LPS   N      BR         6
                                         LO         6
                       ND         N      BR         6
                                         LO         6
                                  NN     BR         2
                                         LO         2
                       Transient  N      BR         6
                                         LO         6
                                  NN     BR         2
                                         LO         2
Longitudinal Crack     0.18 LPS   N      BR         6
                                         LO         6
                       0.47 LPS   N      BR         6
                                         LO         6
                       ND         N      BR         6
                                         LO         6
                                  NN     BR         2
                                         LO         2
                       Transient  N      BR         6
                                         LO         6
                                  NN     BR         2
                                         LO         2
No-leak                0.18 LPS   N      BR         6
                                         LO         6
                       0.47 LPS   N      BR         6
                                         LO         6
                       ND         N      BR         6
                                         LO         6
                                  NN     BR         2
                                         LO         2
                       Transient  N      BR         6
                                         LO         6
                                  NN     BR         2
                                         LO         2
Orifice Leak           0.18 LPS   N      BR         6
                                         LO         6
                       0.47 LPS   N      BR         6
                                         LO         6
                       ND         N      BR         6
                                         LO         6
                                  NN     BR         2
                                         LO         2
                       Transient  N      BR         6
                                         LO         6
                                  NN     BR         2
                                         LO         2

### Signal quality summary

                        samples                                 fs_hz                             duration_s                       nan_count          constant             duplicate_count                        outlier_count_robust                  
                            min      max           mean           min           max          mean        min        max       mean       min max mean      min    max mean             min     max           mean                  min   max        mean
sensor_type                                                                                                                                                                                                                                             
Accelerometer            869713  1048572  928497.787500  25601.638503  25601.638503  25601.638503  33.970990  40.957222  36.267124         0   0  0.0    False  False  0.0          782945  980076  888993.450000                    0  2547   98.075000
Dynamic Pressure Sensor  900909  1048572  938592.437500  25601.638503  25601.638503  25601.638503  35.189506  40.957222  36.661421         0   0  0.0    False  False  0.0          680055  998764  850034.500000                    0  6295  583.550000
Hydrophone               279460   490460  308816.557377   8000.000000   8000.000000   8000.000000  34.932500  61.307500  38.602070         0   0  0.0    False  False  0.0          239483  463000  279937.147541                    0   447   10.532787

## Feasibility

| Goal | Status | Evidence |
|---|---|---|
| Leak/no-leak and leak type | FEASIBLE | Explicit NL/LC/CC/GL/OL labels |
| Localization | NOT FEASIBLE | No leak-position or sensor-position labels; TDE requires pipe geometry/speed and a true paired arrival event |
| Leak size/flow | NOT FEASIBLE | Flow tokens are background flow conditions, not leak rate/size |
| RUL/predictive failure | NOT FEASIBLE | Controlled 30-second recordings contain no degradation/time-to-failure label |
| No-leak autoencoder | FEASIBLE | NL files are explicit normal examples; validation/test are held out by signal group |

## Known limitations
Hydrophone files are integer RAW streams interpreted as little-endian int32 at 8000 Hz because their byte length and dataset format are consistent with 30 seconds at 8 kHz. CSV time columns determine accelerometer/pressure sampling rates. The audit flags this interpretation for confirmation against the publisher script if one is later obtained.
