# Example invocations

All examples use `elementload.cmd` (Windows); replace with `python -m elementload`
or `elementload.ps1` or `elementload` if installed (see README).

## Quick analysis

```cmd
:: Simple telescoping element from --tube specs
elementload.cmd --tube 0.030,0.002,0.6 --tube 0.025,0.002,0.6 ^
  --tube 0.020,0.002,0.6 --tube 0.016,0.0015,0.6 ^
  --overlap 0.05 --joint rigid --support center:fixed --wind 40

:: Same layout using --extent (per-side coordinates, innermost first)
elementload.cmd --extent 0.030,0.002,0.0,0.3 --extent 0.025,0.002,0.25,0.55 ^
  --extent 0.020,0.002,0.5,0.8 --extent 0.016,0.0015,0.75,1.05 ^
  --joint rigid --support center:fixed --wind 40
```

## .stm antenna file

```cmd
:: Analyse every element in a YagiMechanical file
elementload.cmd --stm data/a1.stm

:: Single element, with graphical output
elementload.cmd --stm data/a1.stm --element "40m RE" --plot stress

:: With ES supports
elementload.cmd --stm data/a1.stm --element "20m RE" --wind 36.6
```

## Optimize (stock element → catalog-optimised chain)

```cmd
:: Fixed-geometry optimiser: keeps the stock tube count/lengths,
:: swaps each for the lightest catalogue tube that survives the wind
elementload.cmd --stm data/a1.stm --element "20m RE" --optimize --wind 36.6

:: Custom catalogue file
elementload.cmd --stm data/mechanical2.stm --element "15m DE" --optimize ^
  --catalog my_tubes.csv

:: Snap per-side lengths to 5 mm increments
elementload.cmd --stm data/a1.stm --element "40m RE" --optimize --length-step 0.005

:: Override material for .stm elements
elementload.cmd --stm data/a1.stm --element "20m RE" --optimize ^
  --material al6060-t66
```

## Layout search (find optimal tube count + taper + sleeve)

```cmd
:: Brute-force tube count, taper and sleeve; prints Pareto frontier
elementload.cmd --stm data/a1.stm --element "20m RE" --optimize --layout-search

:: Narrow tube-count range (faster grid)
elementload.cmd --stm data/a1.stm --element "40m RE" --optimize --layout-search ^
  --layout-n 5 7 --layout-workers 4
```

## Layout search from a bare length (no .stm file)

```cmd
:: Optimal tapering for a 4 m element, catalogue-built reference
elementload.cmd --length 4.0 --optimize --layout-search --wind 40 ^
  --material al6060-t66

:: With custom catalogue and length-step
elementload.cmd --length 6.0 --optimize --layout-search --wind 35 ^
  --catalog my_tubes.csv --length-step 0.005 --layout-n 3 6

:: With explicit supports
elementload.cmd --length 8.0 --optimize --layout-search --wind 45 ^
  --support center:fixed --support 1.5:pin --support -1.5:pin
```

## ES support sweep

```cmd
:: Sweep the ES pin over the element half-length
elementload.cmd --stm data/a1.stm --element "40m RE" --optimize ^
  --es-sweep --wind 35

:: Custom sweep range and resolution
elementload.cmd --stm data/a1.stm --element "40m RE" --optimize ^
  --es-sweep --es-range 3 10 --es-steps 20 --wind 35
```

## Plot

```cmd
:: Coloured by stress (default)
elementload.cmd --stm data/a1.stm --element "40m RE" --plot stress

:: Other quantities
elementload.cmd --stm data/a1.stm --element "20m RE" --plot moment ^
  --plot-file 20m_moment.png
elementload.cmd --stm data/a1.stm --element "20m RE" --plot deflection
```

## Dynamic / environmental options

```cmd
:: Override reference height and terrain roughness
elementload.cmd --stm data/a1.stm --element "40m RE" --height 20 --z0 0.3

:: Damping ratio for dynamic response
elementload.cmd --stm data/a1.stm --element "40m RE" --damping 0.02
```

## ES model selection

```cmd
:: Rigid pin only (guys restrain wind load)
elementload.cmd --stm data/a1.stm --element "40m RE" --es-model rigid

:: Sag rope only (guys restrain self-weight only)
elementload.cmd --stm data/a1.stm --element "40m RE" --es-model sag

:: Both side by side (default)
elementload.cmd --stm data/a1.stm --element "40m RE" --es-model both
```

## Materialised layout search result (re-run a Pareto design)

Each layout-search Pareto design prints a ready-to-paste command:

```cmd
elementload.cmd --extent 0.04,0.002,0.000,0.250 --extent 0.035,0.002,0.150,0.983 ^
  --extent 0.03,0.002,0.883,1.863 --extent 0.025,0.002,1.763,2.918 ^
  --extent 0.02,0.0015,2.818,4.185 --extent 0.012,0.001,4.085,5.760 ^
  --material al6060-t66 --joint rigid --support center:fixed --wind 36.6
```
