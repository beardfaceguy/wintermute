# Fiber Optic Drone Deployment: Running Lines Between Remote Locations

## Concept Overview

Fiber-optic tethered drones (FOG drones) are controlled via a thin fiber optic cable that simultaneously serves as the command/control link and a high-bandwidth data channel. The idea explored here: adapt these platforms to *lay* fiber optic cable as they fly, using the drone itself as the deployment mechanism to string lines between remote locations where traditional ground-based installation is impractical.

---

## Why FOG Drones Are Uniquely Suited for This

Standard drones relay control signals over RF (Wi-Fi, radio), which limits range and is vulnerable to jamming and interference. FOG drones thread out fiber as they fly, which gives them:

- **Unlimited effective range** — no RF propagation loss; signal integrity over kilometers
- **High bandwidth, low latency** — gigabit+ control and sensor data on the tether
- **Interference immunity** — no RF signature, no jamming surface
- **Payload efficiency** — the tether *is* the mission; no separate RF comms hardware needed

This means the tether isn't overhead — it's the product being deployed.

---

## Deployment Approaches

### 1. Point-to-Point Line Stringing
The drone lifts from a ground spool, flies a planned route, and anchors the terminal end at the destination. A lightweight messenger line or the fiber itself is spooled out under controlled tension. Suitable for:
- Canyon or river crossings
- Forest canopy gaps
- Remote repeater-to-repeater links

**Key challenge:** Fiber is fragile under tension and bending stress. Deployment fiber would need to be ruggedized (armored or aerial-rated) and the spool tension management must prevent kinks.

### 2. Relay-Hop Deployment
For longer distances exceeding a single drone's range or flight time, multiple drones fly sequential segments. Each drone anchors its end, then a ground crew or autonomous system splices and continues. Effectively a leapfrog pattern:

```
[Base] --drone1--> [Anchor A] --drone2--> [Anchor B] --drone3--> [Destination]
```

### 3. Overhead Lashing
Rather than laying fiber on the ground, the drone strings it aerially between existing poles, trees, or temporary anchor posts — mimicking aerial fiber installation without a bucket truck or helicopter. This keeps the cable off the ground, reducing wildlife interference and flood risk.

### 4. Hybrid: Temporary + Permanent Links
Deploy a lightweight temporary fiber link (e.g., for emergency comms, disaster response) quickly, then use that link to coordinate a more permanent installation later. The drone-deployed line serves as a working connection during construction of the permanent route.

---

## Technical Considerations

### Fiber Spool Design
- **Micro-fiber cable** (250–900 µm diameter) minimizes weight; armored versions add ~1–2 mm OD
- Spool must apply consistent back-tension — too loose causes tangles, too tight stresses the fiber
- Breakaway anchors at both ends protect the fiber if the drone loses the line
- Estimate: 1 km of standard 900 µm fiber weighs ~0.8–1.2 kg; a 2 km spool is marginal for most commercial FOG drones

### Flight Planning
- Route must avoid sharp bends (bend radius >30 mm for most single-mode fiber)
- Terrain-following flight keeps catenary sag predictable
- Wind loading on a deployed line can pull the drone off course — needs active compensation
- Tree canopy and power line avoidance is critical; LiDAR or pre-planned waypoints required

### Fiber Handling at the Endpoints
- Terminal anchoring needs to be robust: UV-stable, weatherproof anchor hardware
- Splicing in the field (fusion or mechanical) adds time but is well-understood
- Pre-connectorized pigtails on the spool ends speed up termination

### Drone Platform Requirements
- High payload capacity: 2–5 kg useful load for a meaningful fiber spool
- Long endurance: 20–45 min minimum flight time under load
- Autonomous waypoint flight with terrain awareness
- Redundant flight control (the FOG tether provides a backup control path by design)

### Regulatory
- BVLOS (beyond visual line of sight) operations required for meaningful range
- FAA Part 107 waivers (US) or equivalent; some remote/emergency use cases have expedited pathways
- Tethered drone regulations differ by jurisdiction — the deployed line may trigger additional airspace rules

---

## Current State of the Art

| Platform | Tether Type | Max Tether Length | Notes |
|---|---|---|---|
| Elistair Orion 2 | Power + data | 100 m (tethered ops) | Designed for persistent surveillance, not deployment |
| Spiderdyne Systems | Fiber optic | 2+ km | Military focus, FOG control |
| Elbit Systems Skylark | Fiber | ~2 km | ISR; tether is control only |
| Custom DIY FOG builds | Standard SMF | Varies | Research/hobbyist; fragile |

Most commercial FOG drones are designed for the tether to be the control link, *not* to be left in place. Adapting them for leave-behind deployment is a novel operational mode rather than a hardware redesign.

---

## Advantages Over Conventional Remote Fiber Deployment

| Method | Cost | Speed | Terrain Limit | Notes |
|---|---|---|---|---|
| Ground trenching | High | Slow | Severe | Not viable in mountains, wetlands |
| Helicopter stringing | Very high | Moderate | Moderate | Requires FAA coordination, weather dependent |
| Balloon/kite | Low | Slow | Wind-dependent | Unpredictable routing |
| FOG drone deployment | Moderate | Fast | Low | Precise routing, autonomous, repeatable |

---

## Open Questions / Research Directions

1. **Spool tension control algorithms** — PID or model-predictive control for consistent fiber payout under variable wind and terrain
2. **Fiber survivability** — What ruggedization level is needed for a drone-deployed aerial span? What's the expected lifespan without conduit?
3. **Multi-drone coordination** — Can two drones work in tandem (one each end) to string a line faster and with better catenary control?
4. **Emergency/disaster use case** — FEMA, wildfire comms, or military forward operating base connectivity; what are the specific requirements?
5. **Autonomous anchor detection** — Can computer vision identify suitable anchor points (poles, trees, rock faces) along a planned route?
6. **Splice drone** — A secondary drone or ground robot that performs field splicing at relay points without human intervention

---

## Next Steps

- [ ] Survey existing FOG drone platforms for payload and tether-length specs
- [ ] Model spool weight vs. fiber length for 500 m, 1 km, 2 km runs
- [ ] Identify candidate terrain types / use cases with the strongest ROI vs. conventional methods
- [ ] Look into BVLOS waiver pathways for infrastructure deployment category
- [ ] Prototype spool tension control on a bench test rig before flight testing
