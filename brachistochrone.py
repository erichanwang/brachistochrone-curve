import pygame
import numpy as np
import math

# --- Constants ---
WIDTH, HEIGHT = 1200, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brachistochrone Curve Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# Physics
G = 9.81  # Acceleration due to gravity

# Curve parameters
START_POS = (100, 100)
END_X = 1100
END_Y = 700

# --- Curve Functions ---
def linear_func(x):
    """A straight line from START_POS to (END_X, END_Y)"""
    x1, y1 = START_POS
    x2, y2 = END_X, END_Y
    if x2 - x1 == 0:
        return y1
    m = (y2 - y1) / (x2 - x1)
    return m * (x - x1) + y1

def parabola_func(x):
    """A parabola sagging between START_POS and (END_X, END_Y)"""
    x1, y1 = START_POS
    x2, y2 = END_X, END_Y
    
    # Linear component
    m = (y2 - y1) / (x2 - x1)
    linear_y = m * (x - x1) + y1
    
    # Parabolic component for the sag. The coefficient controls the amount of sag.
    # A negative product is needed to sag "down" in pygame's inverted-y coordinates.
    sag_coeff = 0.0007
    parabolic_sag = sag_coeff * (x - x1) * (x - x2)
    
    return linear_y - parabolic_sag

def generate_cycloid_points_no_scipy(x_start, y_start, x_end, y_end, num_points=500):
    """
    Generates points for a cycloid curve using a numerical search to find the
    optimal parameters, removing the need for scipy.
    """
    # Use a numerical search (bisection method) to find the correct theta_end
    def find_theta_end(k, low=1e-6, high=2*math.pi, tol=1e-6, max_iter=100):
        for _ in range(max_iter):
            mid = (low + high) / 2
            if mid == 0 or abs(1 - math.cos(mid)) < 1e-9: # Avoid division by zero
                low = mid
                continue
            
            g_mid = (mid - math.sin(mid)) / (1 - math.cos(mid))
            
            if abs(g_mid - k) < tol:
                return mid
            
            if g_mid < k:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    # Ratio k determines the shape of the cycloid
    if abs(y_end - y_start) < 1e-9: return [] # Avoid division by zero
    k = (x_end - x_start) / (y_end - y_start)
    
    theta_end = find_theta_end(k)
    
    if abs(1 - math.cos(theta_end)) < 1e-9: return [] # Avoid division by zero
    a = (y_end - y_start) / (1 - math.cos(theta_end))
    
    thetas = np.linspace(0, theta_end, num_points)
    points = []
    for theta in thetas:
        x = x_start + a * (theta - np.sin(theta))
        y = y_start + a * (1 - np.cos(theta))
        points.append((x, y))
    return points

# --- Ball Class ---
class Ball:
    def __init__(self, color, path_points):
        self.color = color
        self.path_points = path_points
        self.radius = 10
        
        # Pre-calculate cumulative distances for accurate physics
        self.cumulative_distances = [0.0]
        if len(path_points) > 1:
            for i in range(1, len(path_points)):
                p1 = path_points[i-1]
                p2 = path_points[i]
                dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                self.cumulative_distances.append(self.cumulative_distances[-1] + dist)
        
        self.reset()

    def reset(self):
        self.pos_index = 0
        self.x, self.y = self.path_points[0]
        self.v = 0
        self.finished = False
        self.distance_traveled = 0.0
        self.start_time = None

    def update(self, dt):
        if self.finished:
            return

        # If the ball is at the start, give it a nudge to begin moving.
        if self.pos_index == 0 and len(self.path_points) > 1:
            self.pos_index = 1
            self.x, self.y = self.path_points[self.pos_index]

        if self.start_time is None:
            self.start_time = pygame.time.get_ticks()

        # Use energy conservation: v = sqrt(2*g*h)
        start_y = self.path_points[0][1]
        current_h = self.y - start_y
        
        # The velocity in pixels per second. A scaling factor is used for visualization speed.
        # Let's make the physics more realistic. 100 pixels = 1 meter.
        PIXELS_PER_METER = 100
        G_IN_PIXELS = G * PIXELS_PER_METER

        if current_h > 0:
            # v = sqrt(2 * g_pixels * h_pixels)
            self.v = math.sqrt(2 * G_IN_PIXELS * current_h)
        else:
            self.v = 0

        # Update distance traveled based on velocity and time step
        self.distance_traveled += self.v * dt

        # Find the new position on the path based on distance traveled
        # This is more accurate than moving by a fixed number of indices
        new_pos_index = self.pos_index
        while (new_pos_index < len(self.cumulative_distances) and
               self.cumulative_distances[new_pos_index] < self.distance_traveled):
            new_pos_index += 1
        
        self.pos_index = min(new_pos_index, len(self.path_points) - 1)
        self.x, self.y = self.path_points[self.pos_index]

        # Check if finished
        if self.pos_index >= len(self.path_points) - 1:
            self.finished = True
            finish_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
            print(f"{self.color} finished in {finish_time:.2f} seconds")


    def draw(self, win):
        pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), self.radius)
        if self.finished:
             font = pygame.font.SysFont(None, 24)
             text = font.render("Finished!", True, self.color)
             win.blit(text, (self.x + 20, self.y - 10))

# --- Main Function ---
def main():
    pygame.init()
    run = True
    clock = pygame.time.Clock()

    # Generate paths
    num_points = 500
    x_coords = np.linspace(START_POS[0], END_X, num_points)
    
    linear_path = list(zip(x_coords, [linear_func(x) for x in x_coords]))
    parabola_path = list(zip(x_coords, [parabola_func(x) for x in x_coords]))
    
    cycloid_path = generate_cycloid_points_no_scipy(START_POS[0], START_POS[1], END_X, END_Y, num_points)

    # --- Diagnostics ---
    print(f"Generated {len(linear_path)} points for linear path.")
    print(f"Generated {len(parabola_path)} points for parabola path.")
    print(f"Generated {len(cycloid_path)} points for cycloid path.")

    # Create balls for each path, only if the path is not empty
    ball_linear = Ball(RED, linear_path) if linear_path else None
    ball_parabola = Ball(GREEN, parabola_path) if parabola_path else None
    ball_cycloid = Ball(BLUE, cycloid_path) if cycloid_path else None

    balls = [ball for ball in [ball_linear, ball_parabola, ball_cycloid] if ball]
    if not balls:
        print("Error: No valid paths were generated. Cannot run simulation.")
        return
    
    started = False

    def draw_window():
        WIN.fill(WHITE)
        
        # Draw paths
        if len(linear_path) > 1:
            pygame.draw.lines(WIN, GRAY, False, linear_path, 5)
        if len(parabola_path) > 1:
            pygame.draw.lines(WIN, GRAY, False, parabola_path, 5)
        if cycloid_path and len(cycloid_path) > 1:
            pygame.draw.lines(WIN, GRAY, False, cycloid_path, 5)

        # Draw balls
        for ball in balls:
            ball.draw(WIN)

        # Draw labels
        font = pygame.font.SysFont(None, 36)
        win_text = font.render("Press SPACE to start/reset", True, BLACK)
        WIN.blit(win_text, (10, 10))
        
        if linear_path:
            label_linear = font.render("Linear", True, RED)
            WIN.blit(label_linear, (linear_path[50][0], linear_path[50][1] - 30))
        
        if parabola_path:
            label_parabola = font.render("Parabola", True, GREEN)
            WIN.blit(label_parabola, (parabola_path[50][0], parabola_path[50][1] - 30))
        
        if cycloid_path:
            label_cycloid = font.render("Brachistochrone (Cycloid)", True, BLUE)
            WIN.blit(label_cycloid, (cycloid_path[50][0], cycloid_path[50][1] - 30))

        pygame.display.update()

    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    for ball in balls:
                        ball.reset()
                    started = True

        if started:
            dt = clock.get_time() / 1000.0  # Time since last frame in seconds
            for ball in balls:
                ball.update(dt)

        draw_window()

    pygame.quit()

if __name__ == "__main__":
    main()
