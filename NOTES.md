magnets on both sides of pcb
stacked pcbs
rectangular magnet
axial cutouts for air flow
laminated electrical steel or similar in center of coils
corner tricks to minimize 'wasted' copper
built in dc fan that doesn't change direction with motor (pin on driver board to enable/disable - only enable when current is highish)
fan is directly on top / built onto driver board (cools driver and motor!)

1oz copper = 1.37 mils 

Min trace width and spacing: 0.1mm / 4mil (4 layer, 1oz)

~1.2mm for 4 layer = 47.2mil

Min 1oz for outer, 0.5oz for inner

2 * (1 + 0.5) = 3oz total = 4.11mil / 47.2mil = 8.7%

8.7% * 0.5 = 4.3% packing eff

24ga = 0.51mm diam, best: 90%, typ: 75% packing eff.

Hallbach array?

More turns isn't always better, it's current density that's important. But heating limits current density

Stacking pcb and magnet rotor layers!! only enabled by pcb coils

try:
stacking (double magnets means double strength, double pcb means double torque but half speed, more inductance)
more turns vs current density
capping ends with hallbach array

water cooling possible?!
- conformal coating pcbs
- off center fitting on either end of motor
- slits needed in both rotors and stators

Smaller magnets can be used for horizontal parts of hallbach array!

No central water supply needed, just some way of radiating / cooling with air

PCBs probably don't need conformal coating if clamped hard enough,
and all exposed copper is far from center

This same thing could be a great generator, just stack more inductance

Stack with castellated end pads + jumper connections

Can test kV by spinning with drill or with tachometer and powering at known voltage

Flatness of PCB allows less air gap

Ultimate version might be: 2x hallbach array on ends, straight magnets in center, 2 spots for pcbs to go

Stators to hold magnets can be printed for now, could be waterjet later

Eventually centers of coils can have both iron/steel and water flowing

Try filling center of coil with copper

Try spinning pcb instead of magnets to greatly decrease rotational inertia (acceleration)

Test wye vs delta (make configurable for either) but wye seems way to go

To swap stator and rotor, need slip ring fitting + swap shaft keying

Any level of parallel would mean a lot more current but maybe workable with good cooling

Shaft
- 5mm D shaft
- D shaft couplers
- 5mm ID bearing(s)

Use some kind of metal collar to space rotors

Aim for 1mm airgap

Build bearing into 3d printed stators? Spurs with holes around center for air flow (only matters for pcb rotor)

3D printed cone shape in middle of pcb to direct airflow coming from above and below across pcb

Rim driven fans hard or impossible to find - use 2-3 normal angled smaller fans instead
Fan with hub gap would be a bit easier to make, could drive with oring and small motor

PCB rotor
- Easier cooling
- Smaller mechanical stackup (bearings in stators)
- Much less rotational inertia
- Need slip ring (losses probable) for phases AND encoder (or just integrate driver on board BLE?)
PCB stator
- Either need single bearing (larger airgap) or larger mechanical stackup
- No slip ring, adding more stuff to pcb is easy

Single rotor, double stator vs double rotor, single stator