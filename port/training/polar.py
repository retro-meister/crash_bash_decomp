import math


PARAMETER_BOUNDS = ((650, 2050), (400, 1240), (0, 9), (0, 15), (550, 1500), (0.55, 0.99), (0.15, 1.1), (250, 1200))
CHARGING = 0x51


def steer(x, z):
    return 1 + round(math.atan2(x, z) / (math.pi / 4)) % 8 if x or z else 0


def action(result, parameters):
    radius, reach, lead, brake, travel, attack_cos, aggression, dodge = parameters
    state = result["frames"][-1]["polar"]
    x, z, vx, vz, angle, charge, mode = state[0][:7]
    rivals = [opponent for opponent in state[1:] if opponent[7]]
    if not rivals or mode == CHARGING:
        return 0
    projected_x, projected_z = x + vx * brake, z + vz * brake
    if math.hypot(projected_x, projected_z) > radius:
        return steer(-projected_x, -projected_z)
    closest = min(rivals, key=lambda p: (p[0] - x) ** 2 + (p[1] - z) ** 2)
    target_x = closest[0] + closest[2] * lead
    target_z = closest[1] + closest[3] * lead
    dx, dz = target_x - x, target_z - z
    distance = math.hypot(dx, dz)
    heading = angle * math.tau / 4096
    fx, fz = -math.sin(heading), -math.cos(heading)
    if 0 < distance < reach and charge >= 180:
        along = (fx * dx + fz * dz) / distance
        if along > attack_cos and math.hypot(x + fx * travel, z + fz * travel) < radius + 300:
            return 9
    if closest[6] == CHARGING and distance < dodge:
        if (x + dz) ** 2 + (z - dx) ** 2 < (x - dz) ** 2 + (z + dx) ** 2:
            return steer(dz, -dx)
        return steer(-dz, dx)
    target_x *= aggression
    target_z *= aggression
    target_radius = math.hypot(target_x, target_z)
    if target_radius > radius - 200:
        target_x *= (radius - 200) / target_radius
        target_z *= (radius - 200) / target_radius
    return steer(target_x - x - vx * brake, target_z - z - vz * brake)
