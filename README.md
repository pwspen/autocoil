# TODO
- [ ] cleanup code & remove unused/redundant (first with pcb_json, then kicad_funcs based on pcb_json contents)
- [x] move outer vias to outside
- [xx] generate phase interconnects on outside (can just connect every coil in phase then delete connection manually)
- [x] generate air gaps in coil center based on innermost points / vias
- [x] generate outside profile
 - [x] circle
 - [x] tab for electrical connection that can be rotated by one or two poles to allow easily stacking boards
 - [x] 3 mounting holes - 2 connect each stacked board