# Laser Paw Patrol

Laser Paw Patrol is an arcade stealth game built with Pygame and OpenCV where you play as a cat sneaking through a ten-level security vault to steal a golden fish bowl.

## How It Works

The main screen renders a first-person cat perspective looking down a three-lane corridor. As the cat runs forward, the camera bounces along with its steps while showing its front paws, whiskers, and ears around the edges of the screen. Alongside the main view, a smaller window on the right side of the screen runs a live 2D side-scrolling version of the same hallway so you can see your position relative to upcoming obstacles from the side.

Above the 2D side window is a live webcam feed. OpenCV tracks your movement in real time. Jumping in front of your camera makes the cat leap over low floor lasers. Crouching down makes the cat tuck into a ball and roll underneath overhead laser gates. Stepping left or right moves the cat across the three lanes of the corridor to dodge security pylons.

Every few seconds, an alarm sounds and a full laser grid sweeps across the room. When the sweep starts, you have to freeze in place until the scan finishes and the lasers turn green. If the frame difference from your webcam goes over the motion threshold while the red lasers are active, the guard dog catches you and the run ends. If no webcam is plugged in, the game switches to a built-in simulation mode so you can still play using the keyboard.

## The Ten Levels

The game is split into ten progressively faster sectors:

1. Lobby Vents
2. Service Corridor
3. Server Hallway
4. Cryo Storage
5. Security Hub
6. Biolab Tunnel
7. Reactor Walkway
8. Executive Floor
9. Diamond Atrium
10. The Golden Vault

Each level increases the length of the hallway, speeds up the cat, shortens the gap between obstacles, and makes the red laser freeze sweeps last longer. You can start from Level 1 and work your way through all ten in a single run, or pick any starting level from the menu using the number keys.

## Controls

Besides moving in front of your webcam, you can use the keyboard at any time:

- Up Arrow, W, or Space makes the cat jump over floor lasers.
- Down Arrow or S makes the cat crouch and roll under high laser gates.
- Left and Right Arrows (or A and D) shift between the left, center, and right lanes.
- Number keys 1 through 9 (and 0 for Level 10) or the Minus and Equals keys select which level to play.
- C calibrates your current standing or sitting posture on the webcam.
- M toggles between live webcam tracking and simulation mode.
- Left and Right square brackets adjust the motion sensitivity threshold for the freeze sweeps.
- Tab opens the local leaderboard and run stats stored in SQLite.
- Enter or R starts a new run, and Escape returns to the menu.
