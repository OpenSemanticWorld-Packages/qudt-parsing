# QUDT parsing

## todo parsing
 - [x] Units (qudt:Unit, qudt: DerivedUnit, qudt:CountingUnit, qudt:ContextualUnit, qudt:AtomicUnit,
   qudt:DimensionlessUnit, qudt:LogarithmicUnit, qudt:CurrencyUnit)
   - Prefixed units
   - Non-Prefixed units
   - Composed units
   - Base units (non-composed)
 - [x] Quantities (qudt:QuantityKind)
   - Fundamental (posses no attribute broader but is pointed at)
   - Non-Fundamental (posses attribute broader)
 - [x] Prefixes (qudt:Prefix 33, qudt:DecimalPrefix 25, qudt:BinaryPrefix 8)
 - [ ] Dimension vector (qudt:QuantityKindDimensionVector / qudt:ISO-DimensionVector)
 - [ ] Constants (qudt:PhysicalConstant)
 - [ ] System of units (qudt:SystemOfUnits) / System of quantities (qudt:SystemOfQuantityKinds)
 - [ ] Coordinate systems (qudt:MarsCoordinateSystem, qudt:LocalCoordinateSystem, qudt:VehicleCoordinateSystem,
   qudt:EarthCoordinateSystem, qudt:ThreeBodyRotatingCoordinateSystem, qudt:LunarCoordinateSystem)

## todo casting
 - [x] Prefixes
