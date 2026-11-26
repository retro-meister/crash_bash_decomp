#include "../../platform/training.h"
#include "../../platform/platform.h"
#include "bash/arkenoid_ovl.h"
#include "bash/main.h"
#include "bash/polar.h"
#include "bash/polar_ovl.h"

static void polar_training_terrain(TrainingFrame *frame) {
    static const int directions[TRAINING_EDGE_RAYS][2] = {
        {0, 256}, {181, 181}, {256, 0}, {181, -181}, {0, -256}, {-181, -181}, {-256, 0}, {-181, 181}
    };
    const SLONG_XYZ origin = Heroes[0].hero_eng->XYZ;
    int distances[TRAINING_OBSTACLES] = {0};
    int on_ice = CheckMelted(&Heroes[0].hero_eng->XYZ, 0);
    int buttons;
    int i;

    training_pad(0, &buttons);
    frame->square_held = (buttons & PAD_SQUARE) != 0;
    for (i = 0; i < TRAINING_EDGE_RAYS; i++) {
        int lo = 0, hi = on_ice ? 4096 : 0;
        while (hi - lo > 32) {
            const int d = (lo + hi) / 2;
            SLONG_XYZ point = origin;
            point.x += d * directions[i][0] / 256;
            point.z += d * directions[i][1] / 256;
            if (CheckMelted(&point, 0)) {
                lo = d;
            } else {
                hi = d;
            }
        }
        frame->edge[i] = lo;
    }
    frame->obstacle_count = 0;
    for (i = 0; i < 80 && PolarIceObjects[i] != NULL; i++) {
        const subengine *object = PolarIceObjects[i];
        int dx, dz, distance, slot;
        if (!(object->flags & 0x8000)) {
            continue;
        }
        dx = object->XYZ.x - origin.x;
        dz = object->XYZ.z - origin.z;
        distance = dx * dx + dz * dz;
        slot = frame->obstacle_count;
        if (slot == TRAINING_OBSTACLES) {
            if (distance >= distances[slot - 1]) {
                continue;
            }
            slot--;
        } else {
            frame->obstacle_count++;
        }
        while (slot > 0 && distance < distances[slot - 1]) {
            distances[slot] = distances[slot - 1];
            frame->obstacles[slot][0] = frame->obstacles[slot - 1][0];
            frame->obstacles[slot][1] = frame->obstacles[slot - 1][1];
            slot--;
        }
        distances[slot] = distance;
        frame->obstacles[slot][0] = dx;
        frame->obstacles[slot][1] = dz;
    }
}

static void polar_training_snapshot(TrainingFrame *frame) {
    int i;

    frame->ice_radius = PolarIceRadius;
    frame->clock = Clock_Time;
    frame->winner = -1;
    for (i = 0; i < TRAINING_PLAYERS; i++) {
        const HeroInfo *hero = &Heroes[i];
        const EXTengine *engine = hero->hero_eng;
        const Ext20 *polar = FindExtension(hero->hero, 0x20000000);
        int *state = frame->polar[i];

        frame->score[i] = !(hero->hero_status & 0xA000);
        if (hero->hero_status & 0x200) {
            frame->winner = i;
        }
        state[0] = engine->XYZ.x;
        state[1] = engine->XYZ.z;
        state[2] = hero->hero_move ? hero->hero_move->x_vel : 0;
        state[3] = hero->hero_move ? hero->hero_move->z_vel : 0;
        state[4] = hero->hero_move ? hero->hero_move->forward_angle & 0xFFF : 0;
        state[5] = polar ? polar->val : 0;
        state[6] = hero->hero_mode;
        state[7] = frame->score[i];
        state[8] = hero->hero_name;
        state[9] = hero->hero_ai ? hero->hero_ai->AIpad : 0;
        state[10] = hero->hero_mode == 0x51 && hero->hero_anim ? hero->hero_anim->Aindex : 0;
        state[11] = hero->hero_frozen;
        state[12] = polar && polar->val >= PolarHeroAttackPower(i);
        state[13] = hero->hero_move ? hero->hero_move->mass : 0;
    }
    polar_training_terrain(frame);
}

int training_snapshot(TrainingFrame *frame, int arena) {
    NAS *ball;
    int i;

    if (GameMode != arena) {
        return 0;
    }
    for (i = 0; i < TRAINING_PLAYERS; i++) {
        const HeroInfo *hero = &Heroes[i];

        if (hero->hero_eng == NULL) {
            return 0;
        }
        frame->score[i] = hero->hero_score;
        frame->position[i] = i < 2 ? hero->hero_eng->XYZ.x : hero->hero_eng->XYZ.z;
        frame->kick_ready[i] = hero->hero_attack_timer == 0;
    }
    frame->round_active = WinCounterState == 4;
    if (arena == TRAINING_POLAR_PANIC_MODE) {
        polar_training_snapshot(frame);
        return 1;
    }
    frame->ball_count = 0;
    ball = HeadOfBalls;
    for (i = 0; ball != NULL && i < TRAINING_BALLS; i++, ball = ball->Group.Child) {
        const EXTobject *object = (const EXTobject *)ball->FUX;
        const EXTengine *engine = ball->Engine;

        if (object != NULL && engine != NULL && object->status == 1) {
            int index = frame->ball_count++;

            frame->balls[index][0] = engine->XYZ.x;
            frame->balls[index][1] = engine->XYZ.z;
        }
    }
    return 1;
}
