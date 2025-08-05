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
    """A parabola passing through START_POS and (END_X, END_Y)"""
    x1, y1 = START_POS
    x2, y2 = END_X, END_Y
    # y = a(x-h)^2 + k, vertex at (h, k)
    # For simplicity, let's make the curve pass through the points
    # y = ax^2 + bx + c is too complex, let's use a simpler form
    # y = k * (x - x1) ^ 2 + y1, find k such that it passes through (x2, y2)
    if (x2 - x1)**2 == 0:
        return y1
    k = (y2 - y1) / ((x2 - x1) ** 2)
    return k * (x - x1) ** 2 + y1


def brachistochrone_func(x):
    """The brachistochrone curve (a cycloid)"""
    x1, y1 = START_POS
    x2, y2 = END_X, END_Y
    a = (y2 - y1) / np.pi  # Radius of the generating circle
    theta = np.arccos(1 - (x - x1) / a)
    return a * (theta - np.sin(theta)) + y1

def get_cycloid_points(x_start, y_start, x_end, y_end, num_points=500):
    """Generates points for a cycloid curve."""
    # Find the optimal radius 'a' for the cycloid connecting the points
    # This is a transcendental equation, so we solve it numerically.
    from scipy.optimize import fsolve

    def find_a(a):
        theta_end = fsolve(lambda theta: a * (theta - np.sin(theta)) - (x_end - x_start), np.pi)[0]
        return a * (1 - np.cos(theta_end)) - (y_end - y_start)

    a_initial_guess = (y_end - y_start) / 2
    a = fsolve(find_a, a_initial_guess)[0]
    
    theta_end = fsolve(lambda theta: a * (theta - np.sin(theta)) - (x_end - x_start), np.pi)[0]
    
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
        self.reset()

    def reset(self):
        self.pos_index = 0
        self.x, self.y = self.path_points[0]
        self.v = 0
        self.finished = False

    def update(self):
        if self.finished:
            return

        # Nudge the ball to start moving if it's at the beginning
        if self.pos_index == 0:
            self.pos_index += 1

        # Get current and next point on the path
        if self.pos_index + 1 >= len(self.path_points):
            self.finished = True
            return

        p1 = self.path_points[self.pos_index]
        p2 = self.path_points[self.pos_index + 1]

        # Calculate distance and height difference between points
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        ds = math.sqrt(dx**2 + dy**2)

        if ds == 0: # Should not happen with good path data
            self.pos_index += 1
            return

        # Use energy conservation: 0.5*m*v^2 = m*g*h => v = sqrt(2*g*h)
        # h is the total vertical distance from the start
        start_y = self.path_points[0][1]
        current_h = self.y - start_y
        self.v = math.sqrt(2 * G * current_h * 50) # Scaling factor for visualization

        # Move the ball along the path based on its velocity
        # This is a simplification. A more accurate way involves time steps.
        # Let's try a time-based approach.
        # v_f^2 = v_i^2 + 2*a*d. Here a = g*sin(alpha)
        
        # Simplified time-based movement
        if self.v > 0:
            # Calculate how many path segments we can travel in one frame
            # This is a simplification to make it look right without a full physics engine
            steps_to_move = int(self.v / 150) # Heuristic value
            self.pos_index += max(1, steps_to_move)

            if self.pos_index >= len(self.path_points):
                self.pos_index = len(self.path_points) - 1
                self.finished = True
            
            self.x, self.y = self.path_points[self.pos_index]


    def draw(self, win):
        pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), self.radius)
        if self.finished:
             font = pygame.font.SysFont(None, 24)
             text = font.render("Finished!", True, self.color)
             win.blit(text, (self.x + 15, self.y - 15))


# --- Main Function ---
def main():
    pygame.init()
    run = True
    clock = pygame.time.Clock()

    # Generate paths
    num_points = 500
    x_coords = np.linspace(START_POS[0], END_X, num_points)
    
    # 1. Straight Line Path
    linear_path = list(zip(x_coords, [linear_func(x) for x in x_coords]))
    
    # 2. Parabola Path
    parabola_path = list(zip(x_coords, [parabola_func(x) for x in x_coords]))

    # 3. Cycloid (Brachistochrone) Path
    try:
        cycloid_path = get_cycloid_points(START_POS[0], START_POS[1], END_X, END_Y, num_points)
    except ImportError:
        print("Scipy not found. Cycloid curve will not be generated.")
        print("Please install it using: pip install scipy")
        cycloid_path = [] # Fallback

    # Create balls for each path
    ball_linear = Ball(RED, linear_path)
    ball_parabola = Ball(GREEN, parabola_path)
    ball_cycloid = Ball(BLUE, cycloid_path) if cycloid_path else None

    balls = [ball for ball in [ball_linear, ball_parabola, ball_cycloid] if ball]
    
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
        
        label_linear = font.render("Linear", True, RED)
        WIN.blit(label_linear, (linear_path[50][0], linear_path[50][1] - 30))
        
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
                    # Reset all balls and start the simulation
                    for ball in balls:
                        ball.reset()
                    started = True


        if started:
            for ball in balls:
                ball.update()

        draw_window()

    pygame.quit()

if __name__ == "__main__":
    main()
