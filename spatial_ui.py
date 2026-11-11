import tkinter as tk
from tkinter import ttk
# Import the custom spatial architecture directly from your engine file
from spatial_engine import SpatialCategoryEngine, FastRuntimeRouter

class SpatialGraphApp:
    def __init__(self, root, model_file):
        self.root = root
        self.root.title("Custom 2D Spatial Engine Interface")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e1e")
        
        self.router = FastRuntimeRouter(model_file)
        self.setup_ui_layout()
        self.draw_base_landscape()

    def setup_ui_layout(self):
        """Constructs a hyper-lightweight interface frame matching low RAM limits."""
        # Top Interactive Control Panel
        control_frame = tk.Frame(self.root, bg="#2d2d2d", pady=10)
        control_frame.pack(fill=tk.X, side=tk.TOP)
        
        lbl = tk.Label(control_frame, text="Enter Agent Swarm Prompt:", fg="white", bg="#2d2d2d", font=("Arial", 11, "bold"))
        lbl.pack(side=tk.LEFT, padx=10)
        
        self.prompt_entry = tk.Entry(control_frame, width=50, font=("Arial", 11), bg="#3d3d3d", fg="white", insertbackground="white")
        self.prompt_entry.pack(side=tk.LEFT, padx=10)
        self.prompt_entry.insert(0, "Assign roles inside the tracking agent swarm.")
        
        route_btn = tk.Button(control_frame, text="Fast Route Vectors", command=self.execute_visual_routing, bg="#007acc", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT)
        route_btn.pack(side=tk.LEFT, padx=10)

        # Main Graph Mapping Screen (Tkinter Canvas uses almost 0MB RAM)
        self.canvas = tk.Canvas(self.root, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        
        # Status Readout Footer
        self.status_lbl = tk.Label(self.root, text="System Ready. Standing by for localized parameter sweeps.", bg="#1e1e1e", fg="#aaaaaa", font=("Arial", 10))
        self.status_lbl.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

    def to_canvas_coords(self, x, y):
        """Translates abstract model weight coordinates to screen pixel positions."""
        # Center point of our window
        cx = self.canvas.winfo_width() / 2 if self.canvas.winfo_width() > 1 else 450
        cy = self.canvas.winfo_height() / 2 if self.canvas.winfo_height() > 1 else 300
        # Scale factor to prevent text overlap
        scale = 1.8
        return cx + (x * scale), cy - (y * scale)

    def draw_base_landscape(self):
        """Plots the structural macro-category anchor coordinates on screen."""
        self.root.update()
        self.canvas.delete("all")
        
        # Draw Coordinate Grid lines
        cx, cy = self.to_canvas_coords(0, 0)
        self.canvas.create_line(0, cy, cx*2, cy, fill="#252525", width=1)
        self.canvas.create_line(cx, 0, cx, cy*2, fill="#252525", width=1)
        
        # Plot Category Hubs
        for cat_name, (cx_val, cy_val) in self.router.spatial_index.get("CATEGORIES", {}).items() if hasattr(self.router, 'categories') else [
            ("management", (100.0, 100.0)), ("traction_eos", (120.0, -80.0)),
            ("agent_swarm", (-150.0, 150.0)), ("repository", (-50.0, -120.0)), ("general_text", (0.0, 0.0))
        ]:
            px, py = self.to_canvas_coords(cx_val, cy_val)
            self.canvas.create_oval(px-8, py-8, px+8, py+8, fill="#ff9900", outline="#ffffff", width=1, tags="anchor")
            self.canvas.create_text(px, py-18, text=cat_name.upper(), fill="#ff9900", font=("Arial", 9, "bold"))

        # Plot Individual Local Word Nodes
        for word, metadata in self.router.spatial_index.items():
            wx, wy = metadata["coords"]
            px, py = self.to_canvas_coords(wx, wy)
            self.canvas.create_oval(px-3, py-3, px+3, py+3, fill="#555555")
            self.canvas.create_text(px+6, py, text=word, fill="#888888", font=("Arial", 8), anchor=tk.W)

    def execute_visual_routing(self):
        """Triggers the geometric routing sequence and visually updates the canvas lanes."""
        self.draw_base_landscape() # Reset the view
        prompt = self.prompt_entry.get()
        
        # 1. Compute target destination coordinates using the FTL
        tx, ty = self.router.route_incoming_prompt(prompt)
        t_px, t_py = self.to_canvas_coords(tx, ty)
        
        # 2. Draw target coordinate crosshair
        self.canvas.create_oval(t_px-12, t_py-12, t_px+12, t_py+12, outline="#00ffcc", width=2, tags="route")
        self.canvas.create_line(t_px-20, t_py, t_px+20, t_py, fill="#00ffcc", width=1, tags="route")
        self.canvas.create_line(t_px, t_py-20, t_px, t_py+20, fill="#00ffcc", width=1, tags="route")
        
        # 3. Pull weights within our radius limit from the flash drive
        active_nodes = self.router.extract_nearest_weights(tx, ty, radius=60.0)
        
        # 4. Light up activated file routes on screen
        for active_word in active_nodes.keys():
            if active_word in self.router.spatial_index:
                wx, wy = self.router.spatial_index[active_word]["coords"]
                w_px, w_py = self.to_canvas_coords(wx, wy)
                # Draw high-priority routing lane from cursor to file nodes
                self.canvas.create_line(t_px, t_py, w_px, w_py, fill="#00ffcc", dash=(4, 4), width=1, tags="route")
                self.canvas.create_oval(w_px-4, w_py-4, w_px+4, w_py+4, fill="#00ffcc", outline="#ffffff")
                self.canvas.create_text(w_px+6, w_py, text=active_word, fill="#ffffff", font=("Arial", 9, "bold"), anchor=tk.W)
                
        self.status_lbl.config(
            text=f"FTL Fast Routed to X: {tx:.1f}, Y: {ty:.1f} | Successfully pulled {len(active_nodes)} local text weights from disk storage."
        )

if __name__ == "__main__":
    # Point directly to the binary weight mapping file built by your compiler
    target_binary_map = "dense_spatial_map.bin"
    
    root = tk.Tk()
    app = SpatialGraphApp(root, target_binary_map)
    root.mainloop()
