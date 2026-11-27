#include <math.h>
#include "../../platform/platform.h"
#include "../../platform/training.h"
#include "bash/engine.h"
#include "bash/hero.h"
#include "bash/main.h"
#include "bash/polar.h"
#include "bash/polar_ovl.h"

enum { PLAN_TICKS = 32, WINDUP_TICKS = 6, CHARGE_END = 27, MAX_EDGES = 64 };

typedef struct {
    float x, z;
} Vec;

typedef struct {
    Vec position, velocity, bias, facing;
    float speed, boost, acceleration, friction, mass;
    int windup, lock, frozen, alive;
} Skater;

typedef struct {
    Vec normal;
    float offset;
} Plane;

typedef struct {
    Plane edge[MAX_EDGES];
    Vec obstacle[80];
    int edges, obstacles;
} Arena;

static const float ApproachOffset = 660.0f;
static const float EdgeBuffer = 250.0f;
static const float ChargeRange = 1450.0f;

static const Vec Directions[9] = {
    {0, 0}, {0, 1}, {.7071f, .7071f}, {1, 0}, {.7071f, -.7071f},
    {0, -1}, {-.7071f, -.7071f}, {-1, 0}, {-.7071f, .7071f}
};

static const int Buttons[9] = {
    0, PAD_UP, PAD_UP | PAD_RIGHT, PAD_RIGHT, PAD_RIGHT | PAD_DOWN,
    PAD_DOWN, PAD_DOWN | PAD_LEFT, PAD_LEFT, PAD_LEFT | PAD_UP
};

static Vec add(Vec a, Vec b) { return (Vec){a.x + b.x, a.z + b.z}; }
static Vec sub(Vec a, Vec b) { return (Vec){a.x - b.x, a.z - b.z}; }
static Vec scale(Vec a, float s) { return (Vec){a.x * s, a.z * s}; }
static float dot(Vec a, Vec b) { return a.x * b.x + a.z * b.z; }
static float length2(Vec a) { return dot(a, a); }
static float lower(float a, float b) { return a < b ? a : b; }
static float higher(float a, float b) { return a > b ? a : b; }
static Vec unit(Vec a) { return scale(a, 1.0f / higher(sqrtf(length2(a)), 1.0f)); }

static Vec heading(int angle) {
    return (Vec){sin_cos_table[(angle + 0x400) & 0xFFF].unk2 / 4096.0f,
                 sin_cos_table[(angle + 0x800) & 0xFFF].unk2 / 4096.0f};
}

static void load_arena(Arena *arena) {
    const u8 *map = (const u8 *)MapScene->map;
    const s32 *zones = (const s32 *)(map + 0x14 + ((const map_map *)map)->contour_grids);
    const MapContour *contour = SELF_REL(&zones[1]);
    int i;

    arena->edges = 0;
    if (contour != NULL && contour->nEdges <= MAX_EDGES) {
        const MapEdge *edge = contour->edges;
        arena->edges = contour->nEdges;
        for (i = 0; i < arena->edges; ++i) {
            const int angle = -edge[i].angle;
            const Vec normal = {sin_cos_table[angle & 0xFFF].unk2 / 4096.0f,
                                -sin_cos_table[(angle - 0x400) & 0xFFF].unk2 / 4096.0f};
            arena->edge[i] = (Plane){normal, dot(normal, (Vec){edge[i].p0.x, edge[i].p0.z})};
        }
    }
    arena->obstacles = 0;
    for (i = 0; i < 80 && PolarIceObjects[i] != NULL; ++i) {
        const subengine *object = PolarIceObjects[i];
        if (object->flags & 0x8000) {
            arena->obstacle[arena->obstacles++] = (Vec){object->XYZ.x, object->XYZ.z};
        }
    }
}

static float clearance(const Arena *arena, Vec point) {
    float distance = 4000.0f;
    for (int i = 0; i < arena->edges; ++i) {
        distance = lower(distance, dot(arena->edge[i].normal, point) - arena->edge[i].offset);
    }
    return distance;
}

static Skater load_skater(int player) {
    const HeroInfo *h = &Heroes[player];
    const EXTheromove *move = h->hero_move;
    const int phase = h->hero_mode == 0x51 && h->hero_anim ? h->hero_anim->Aindex : 0;
    const int ice_mass = (move->mass * 3) >> 3;
    Skater s = {
        .position = {h->hero_eng->XYZ.x, h->hero_eng->XYZ.z}, .velocity = {move->x_vel, move->z_vel},
        .facing = heading(move->forward_angle), .speed = PolarHeroMass[player], .boost = PolarHeroThrowPower(player),
        .acceleration = move->x_acc > 0 ? move->x_acc : 2, .friction = move->x_dec > 0 ? move->x_dec : 1,
        .mass = move->mass, .frozen = h->hero_frozen,
        .alive = !(h->hero_status & 0xA000)
    };
    s.bias.x = (ice_mass * sin_cos_table[(PolarTiltZ - 0x400) & 0xFFF].unk2) >> 12;
    s.bias.z = -((ice_mass * sin_cos_table[(PolarTiltX - 0x400) & 0xFFF].unk2) >> 12);
    if (phase > 0) {
        const int start = h->hero_name == 5 || h->hero_name == 6 ? WINDUP_TICKS : 1;
        s.windup = phase <= start ? start + 1 - phase : 0;
        s.lock = CHARGE_END - phase;
    }
    return s;
}

static float toward(float current, float target, float amount) {
    return current + higher(-amount, lower(amount, target - current));
}

static void move_skater(Skater *s, Vec direction) {
    if (s->windup > 0) {
        s->velocity = --s->windup ? (Vec){0, 0} : scale(s->facing, higher(0, s->boost - 1));
    } else {
        if (s->frozen > 0) {
            --s->frozen;
            direction = (Vec){0, 0};
        }
        const Vec target = add(scale(direction, s->speed), s->bias);
        const float friction = s->lock > 0 ? higher(s->friction, 2.0f) : s->friction;
        s->velocity.x = toward(s->velocity.x, target.x, target.x != 0 ? s->acceleration : friction);
        s->velocity.z = toward(s->velocity.z, target.z, target.z != 0 ? s->acceleration : friction);
    }
    if (s->lock > 0) {
        --s->lock;
    }
    s->position = add(s->position, s->velocity);
}

static float collide(Skater *a, Skater *b) {
    const Vec delta = sub(b->position, a->position);
    const float d2 = length2(delta);
    if (d2 <= 0.01f || d2 >= 256.0f * 256.0f) {
        return 0;
    }
    const float d = sqrtf(d2);
    const Vec normal = scale(delta, 1.0f / d);
    const Vec separation = scale(normal, (258.0f - d) * .5f);
    const Vec relative = sub(a->velocity, b->velocity);
    float closing = dot(relative, normal);

    a->position = sub(a->position, separation);
    b->position = add(b->position, separation);
    if (closing <= 0) {
        return 0;
    }
    const float ma = a->mass + (a->lock > 0 ? 544.0f : 0.0f);
    const float mb = b->mass + (b->lock > 0 ? 544.0f : 0.0f);
    if (ma <= 0 || mb <= 0) {
        return 0;
    }
    a->velocity = sub(a->velocity, scale(relative, 2 * mb / (ma + mb)));
    b->velocity = add(b->velocity, scale(relative, 2 * ma / (ma + mb)));
    return closing;
}

static int hit_obstacle(const Arena *arena, Skater *s) {
    for (int i = 0; i < arena->obstacles; ++i) {
        const Vec delta = sub(s->position, arena->obstacle[i]);
        const float distance2 = length2(delta);
        if (distance2 < 331 * 331 && distance2 > 0.01f) {
            const Vec normal = unit(delta);
            const float inward = dot(s->velocity, normal);
            s->position = add(arena->obstacle[i], scale(normal, 333));
            if (inward < 0) {
                s->velocity = scale(sub(s->velocity, scale(normal, 1.6f * inward)), .55f);
                return 1;
            }
        }
    }
    return 0;
}

static Vec recovery(const Arena *arena, const Skater *me, Vec goal) {
    const Vec projected = add(me->position, scale(me->velocity, 10));
    if (clearance(arena, projected) < EdgeBuffer) {
        return unit(scale(projected, -1));
    }
    return unit(sub(goal, projected));
}

static float rollout(const Arena *arena, const Skater *initial, Vec first, int charge, int reacts, int target) {
    Skater players[TRAINING_PLAYERS];
    Vec intent[TRAINING_PLAYERS];
    Skater *me = &players[0];
    float total = 0, risk = 0;
    int tick, i;

    for (i = 0; i < TRAINING_PLAYERS; ++i) {
        players[i] = initial[i];
        intent[i] = unit(initial[i].velocity);
    }
    if (charge) {
        me->windup = WINDUP_TICKS + 1;
        me->lock = CHARGE_END;
    }
    if (charge || me->windup > 0) {
        me->facing = first;
    }
    for (tick = 0; tick < PLAN_TICKS; ++tick) {
        Vec target_pos = players[target].position;
        Vec goal = sub(target_pos, scale(unit(target_pos), ApproachOffset));
        Vec direction = tick < 7 ? first : recovery(arena, me, goal);
        if (me->lock > 0) {
            direction = (Vec){0, 0};
        }
        move_skater(me, direction);
        total -= 18 * hit_obstacle(arena, me);
        for (i = 1; i < TRAINING_PLAYERS; ++i) {
            Skater *opponent = &players[i];
            if (!opponent->alive) {
                continue;
            }
            if (reacts && tick >= 6) {
                const Vec target_pos = clearance(arena, opponent->position) < 420 ? (Vec){0, 0} : me->position;
                intent[i] = unit(sub(target_pos, opponent->position));
            }
            move_skater(opponent, opponent->lock > 0 ? (Vec){0, 0} : intent[i]);
            hit_obstacle(arena, opponent);
            const float closing = collide(me, opponent);
            if (closing > 0) {
                const float outward = dot(unit(opponent->position), opponent->velocity);
                total += .35f * higher(0, outward - 12) - .40f * higher(0, dot(unit(me->position), me->velocity) - 20);
            }
            if (clearance(arena, opponent->position) < -20) {
                opponent->alive = 0;
                total += 610;
            }
        }
        const float safe = clearance(arena, me->position);
        if (safe < 8) {
            return -1380 + 18 * tick + .15f * total;
        }
        risk = higher(risk, EdgeBuffer - safe);
    }
    const Vec opponent = players[target].position;
    const Vec approach = sub(opponent, scale(unit(opponent), ApproachOffset));
    const float stopping_clearance = clearance(arena, add(me->position, scale(me->velocity, 9)));
    total -= .305f * sqrtf(length2(sub(approach, me->position)));
    total -= .75f * risk + 1.84f * higher(0, EdgeBuffer - stopping_clearance);
    if (charge) {
        total -= 60;
    }
    return total;
}

int training_polar_plan(int previous_buttons) {
    const HeroInfo *hero = &Heroes[0];
    const Ext20 *polar;
    Skater initial[TRAINING_PLAYERS];
    Arena arena;
    float best = -INFINITY, nearest = INFINITY;
    int target = 0, best_action = 0, can_charge;
    int i, action;

    if (hero->hero_frozen || !hero->hero_move) {
        return 0;
    }
    for (i = 0; i < TRAINING_PLAYERS; ++i) {
        if (!Heroes[i].hero_move && !(Heroes[i].hero_status & 0xA000)) {
            return 0;
        }
        initial[i] = Heroes[i].hero_move ? load_skater(i) : (Skater){0};
    }
    if (initial[0].lock > 0 && !initial[0].windup) {
        return 0;
    }
    for (i = 1; i < TRAINING_PLAYERS; ++i) {
        if (initial[i].alive) {
            float distance = length2(sub(initial[i].position, initial[0].position));
            if (distance < nearest) {
                target = i;
                nearest = distance;
            }
        }
    }
    if (!target) {
        return 0;
    }
    load_arena(&arena);
    if (arena.edges < 3) {
        return 0;
    }
    polar = FindExtension(hero->hero, 0x20000000);
    can_charge = nearest < ChargeRange * ChargeRange && polar && polar->val >= PolarHeroAttackPower(0) &&
                 !(previous_buttons & PAD_SQUARE) && (hero->hero_mode == 0x49 || hero->hero_mode == 0x4A);
    for (action = initial[0].windup ? 1 : 0; action < (can_charge ? 17 : 9); ++action) {
        const int charge = action > 8;
        const Vec direction = Directions[charge ? action - 8 : action];
        const float steady = rollout(&arena, initial, direction, charge, 0, target);
        const float reacting = rollout(&arena, initial, direction, charge, 1, target);
        const float value = .45f * (steady + reacting) + .10f * lower(steady, reacting);
        if (value > best) {
            best = value;
            best_action = action;
        }
    }
    return best_action > 8 ? Buttons[best_action - 8] | PAD_SQUARE : Buttons[best_action];
}
