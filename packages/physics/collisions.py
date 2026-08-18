"""Deterministic Collision Detection and Resolution for 2D Particles"""

from typing import List, Tuple
from packages.core.models import Particle, Boundary, Vector2D


class BoundaryCollision:
    """Elastic Bounding Box Collision Handler."""
    def __init__(self, restitution: float = 1.0):
        self.restitution = restitution

    def resolve(self, particles: List[Particle], boundary: Boundary) -> List[Particle]:
        updated_particles = []
        for p in particles:
            pos = p.position
            vel = p.velocity
            r = p.radius

            new_x, new_vx = pos.x, vel.x
            new_y, new_vy = pos.y, vel.y

            # X boundary collision
            if pos.x - r < boundary.x_min:
                new_x = boundary.x_min + r
                new_vx = -vel.x * self.restitution
            elif pos.x + r > boundary.x_max:
                new_x = boundary.x_max - r
                new_vx = -vel.x * self.restitution

            # Y boundary collision
            if pos.y - r < boundary.y_min:
                new_y = boundary.y_min + r
                new_vy = -vel.y * self.restitution
            elif pos.y + r > boundary.y_max:
                new_y = boundary.y_max - r
                new_vy = -vel.y * self.restitution

            updated_p = p.with_position(Vector2D(x=new_x, y=new_y)).with_velocity(Vector2D(x=new_vx, y=new_vy))
            updated_particles.append(updated_p)

        return updated_particles


class ParticleCollision:
    """Pairwise 2D Elastic Particle Collision Handler."""
    def __init__(self, restitution: float = 1.0):
        self.restitution = restitution

    def resolve(self, particles: List[Particle]) -> List[Particle]:
        n = len(particles)
        if n < 2:
            return particles

        # Working mutable state arrays for position and velocity
        positions = [p.position for p in particles]
        velocities = [p.velocity for p in particles]

        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = particles[i], particles[j]
                r1, r2 = p1.radius, p2.radius
                pos1, pos2 = positions[i], positions[j]
                vel1, vel2 = velocities[i], velocities[j]

                delta_pos = pos2 - pos1
                dist = delta_pos.norm()
                min_dist = r1 + r2

                if dist < min_dist and dist > 1e-9:
                    # Overlap resolution
                    normal = delta_pos / dist
                    overlap = min_dist - dist
                    # Positional shift proportional to inverse mass
                    total_m = p1.mass + p2.mass
                    shift1 = normal * (overlap * (p2.mass / total_m))
                    shift2 = normal * (overlap * (p1.mass / total_m))
                    pos1 = pos1 - shift1
                    pos2 = pos2 + shift2

                    # Elastic velocity impulse
                    rel_vel = vel2 - vel1
                    vel_normal = rel_vel.dot(normal)

                    # Only resolve if moving towards each other
                    if vel_normal < 0:
                        impulse = (-(1.0 + self.restitution) * vel_normal) / (1.0 / p1.mass + 1.0 / p2.mass)
                        impulse_vector = normal * impulse

                        vel1 = vel1 - (impulse_vector / p1.mass)
                        vel2 = vel2 + (impulse_vector / p2.mass)

                    positions[i] = pos1
                    positions[j] = pos2
                    velocities[i] = vel1
                    velocities[j] = vel2

        updated = []
        for i, p in enumerate(particles):
            updated.append(p.with_position(positions[i]).with_velocity(velocities[i]))
        return updated
