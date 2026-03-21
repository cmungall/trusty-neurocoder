# TODO: EcoSIM Integration

## Status

EcoSIM is cloned at `~/repos/EcoSIM`. Build is in progress (compiling
HDF5/NetCDF from source via cmake). The Fortran code has been read and
key equations extracted for the surrogate in `examples/ecosim_decomp.py`.

## Actual EcoSIM Constants (from Fortran source)

Extracted from the codebase for a faithful reimplementation:

```
RGASC = 8.3143          (J/mol·K, EcoSimConst.F90:30)
OQKI  = 1200.0          (g C/m³, product inhibition constant, NitroPars.F90:211)
DCKM0 = 1000.0          (g C/g soil, substrate Km at zero biomass, NitroPars.F90:220)
DCKI  = 2.5             (inhibition by microbial concentration, NitroPars.F90:203)

SPOSC (specific decomposition rates, MicBGCPars.F90:242-244):
  Litter complexes (K=1-3): protein=7.5, carbohydrate=7.5, cellulose=1.5, lignin=0.5
  Humus (K=4): protein=0.05, others≈0
  Deep humus (K=5): protein=0.05, carbohydrate=0.0167, others=0

TSensGrowth (MicrobMathFuncMod.F90:13-30):
  RTK = 8.3143 * T
  STK = 710.0 * T
  ACTV = 1 + exp((197500-STK)/RTK) + exp((STK-222500)/RTK)
  TSensGrowth = exp(25.229 - 62500/RTK) / ACTV

WatStressMicb (MicBGCMod.F90:790-792):
  Fungi: exp(0.1 * max(PSI, -500))
  Others: exp(0.2 * max(PSI, -500))
```

## Next Steps

### Level 1: Build and run EcoSIM (in progress)

- [ ] Complete EcoSIM build (`./build_EcoSIM.sh`)
- [ ] Run bare_soil example with default config
- [ ] Extract carbon pool trajectories from NetCDF output
- [ ] Train Cajal surrogate on actual EcoSIM output (not reimplementation)

### Level 2: Faithful Python reimplementation

- [ ] Update `examples/ecosim_decomp.py` with actual SPOSC values, OQKI, DCKM0, DCKI
- [ ] Include biomass-dependent Km: `DCKD = DCKM0 * (1 + COQCK/DCKI)`
- [ ] Use litter vs. humus substrate distinction
- [ ] Validate against EcoSIM output once built

### Level 3: Box model driver

- [ ] Use `drivers/boxsbgc/` for a simplified single-layer run
- [ ] Generate trajectories at controlled (T, moisture) conditions
- [ ] Compare Cajal surrogate recovery against box model output

### Level 4: Observational data

- [ ] FLUXNET eddy covariance data for soil respiration
- [ ] SRDB (Soil Respiration Database) measurements
- [ ] Train surrogate on field data; decompiled expressions = empirical findings
