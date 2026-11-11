import struct
import math
from collections import defaultdict, Counter

class SpatialCategoryEngine:
    def __init__(self):
        self.vocabulary = {}
        self.id_to_word = {}
        self.word_coordinates = {}
        self.co_occurrences = defaultdict(Counter)
        
        # Define the structural anchor coordinates for Category Routing
        # This acts as our macro-level semantic map grid
        self.categories = {
            "management":  (100.0, 100.0),
            "traction_eos": (120.0, -80.0),
            "agent_swarm":  (-150.0, 150.0),
            "repository":   (-50.0, -120.0),
            "general_text": (0.0, 0.0)
        }

    def determine_word_category(self, word):
        """Routes specific target keywords to their macro-category anchors."""
        business_terms = {"visionary", "integrator", "scorecard", "meeting", "roles", "hierarchy", "ceo", "board"}
        tech_terms = {"swarm", "agent", "git", "github", "repo", "repository", "code", "inference", "routing"}
        
        if word in business_terms or "traction" in word or "eos" in word:
            return "traction_eos" if "traction" in word else "management"
        elif word in tech_terms:
            return "agent_swarm" if "swarm" in word else "repository"
        return "general_text"

    def ingest_dense_text(self, text):
        """Tokenizes pure prose and registers structural adjacency links."""
        words = [w.strip(".,!?\"()[]{}").lower() for w in text.split() if w.strip()]
        
        # Sliding context window to track relational density
        window_size = 4
        for i, word in enumerate(words):
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    self.co_occurrences[word][words[j]] += 1

    def compile_2d_landscape(self):
        """Calculates exact 2D spatial weights around the category coordinates."""
        print("[Engine] Computing geometric weight coordinates around anchors...")
        all_words = sorted(self.co_occurrences.keys(), key=lambda x: sum(self.co_occurrences[x].values()), reverse=True)
        
        category_counts = Counter()
        for idx, word in enumerate(all_words):
            self.vocabulary[word] = idx
            self.id_to_word[idx] = word
            
            # Find which macro-anchor this word maps near
            cat = self.determine_word_category(word)
            anchor_x, anchor_y = self.categories[cat]
            
            # Cluster words in a micro-spiral outward from their designated category anchor
            count = category_counts[cat]
            angle = count * 0.4
            radius = math.sqrt(count) * 8.0
            
            word_x = anchor_x + (radius * math.cos(angle))
            word_y = anchor_y + (radius * math.sin(angle))
            
            self.word_coordinates[idx] = (word_x, word_y)
            category_counts[cat] += 1

    def save_to_binary_file(self, filename):
        """Compiles the entire landscape into an indexed flat binary file."""
        print(f"[Engine] Writing binary data to disk: {filename}")
        
        # Format Layout:
        # MAGIC SIGNATURE (4 Bytes) | TOTAL RECORDS (4 Bytes)
        # INDEX BLOCKS (For each word): Word (32B) | X (4B Float) | Y (4B Float) | File Pointer (8B Double)
        # DATA MATRIX BLOCKS: Length of links (4B) | [Target ID (4B) + Weight (4B Float) * Length]
        
        with open(filename, "wb") as f:
            f.write(b"SPAT") # Spatial Engine Signifier
            f.write(struct.pack("<I", len(self.vocabulary)))
            
            # Pre-calculate file positions for our spatial lookups
            index_table_size = 8 + (len(self.vocabulary) * (32 + 4 + 4 + 8))
            current_data_offset = index_table_size
            
            index_entries = []
            data_blocks = bytearray()
            
            for word, word_id in self.vocabulary.items():
                wx, wy = self.word_coordinates[word_id]
                links = self.co_occurrences[word]
                
                # Build localized relational weight blocks
                block_buffer = bytearray()
                block_buffer.extend(struct.pack("<I", len(links)))
                for target_word, weight in links.items():
                    target_id = self.vocabulary[target_word]
                    block_buffer.extend(struct.pack("<If", target_id, float(weight)))
                
                index_entries.append((word, wx, wy, current_data_offset))
                data_blocks.extend(block_buffer)
                current_data_offset += len(block_buffer)
                
            # Stream out the master coordinate index
            for word, x, y, offset in index_entries:
                padded_word = word.encode('utf-8')[:32].ljust(32, b'\x00')
                f.write(struct.pack("<32sffd", padded_word, x, y, float(offset)))
                
            # Stream out raw block data arrays
            f.write(data_blocks)
        print("[Engine] Local compilation sequence finalized successfully.")

class FastRuntimeRouter:
    def __init__(self, model_file):
        self.model_file = model_file
        self.spatial_index = {}
        self.load_index_table()
        
    def load_index_table(self):
        """Loads only the structural coordinate pointers into memory (Few Kilobytes)."""
        with open(self.model_file, "rb") as f:
            if f.read(4) != b"SPAT":
                raise ValueError("Incompatible model format layout.")
            total_records = struct.unpack("<I", f.read(4))[0]
            
            for _ in range(total_records):
                word_bytes, x, y, offset = struct.unpack("<32sffd", f.read(48))
                word = word_bytes.decode('utf-8').strip('\x00')
                self.spatial_index[word] = {"coords": (x, y), "offset": offset}

    def route_incoming_prompt(self, prompt_text):
        """Geometrically maps incoming sentence vectors directly to flash drive regions."""
        tokens = [t.strip(".,!?\"").lower() for t in prompt_text.split() if t.strip()]
        
        sum_x, sum_y, valid_hits = 0.0, 0.0, 0
        for token in tokens:
            if token in self.spatial_index:
                sum_x += self.spatial_index[token]["coords"][0]
                sum_y += self.spatial_index[token]["coords"][1]
                valid_hits += 1
                
        if valid_hits == 0:
            return 0.0, 0.0 # Default center fallback
            
        # Target coordinate center point of the incoming sentence
        return (sum_x / valid_hits), (sum_y / valid_hits)

    def extract_nearest_weights(self, target_x, target_y, radius=50.0):
        """Seeks straight to disk locations matching the computed spatial destination."""
        activated_knowledge = {}
        
        # Scan coordinates to pick target index entries within our geometric threshold
        with open(self.model_file, "rb") as f:
            for word, metadata in self.spatial_index.items():
                wx, wy = metadata["coords"]
                distance = math.sqrt((wx - target_x)**2 + (wy - target_y)**2)
                
                if distance <= radius:
                    # Micro-buffer disk seek: Load ONLY this sub-graph node
                    f.seek(int(metadata["offset"]))
                    link_count = struct.unpack("<I", f.read(4))[0]
                    
                    connections = []
                    for _ in range(link_count):
                        target_id, weight = struct.unpack("<If", f.read(8))
                        connections.append((target_id, weight))
                    activated_knowledge[word] = connections
                    
        return activated_knowledge

# ==========================================
# EXECUTION SIMULATION LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Ingest clean, highly dense prose (E.g., Traction / EOS Framework Data)
    compiler = SpatialCategoryEngine()
    
    dense_data = (
        "The Visionary routes organizational trajectory while the Integrator executes operational systems. "
        "A swarm of agents structures a hierarchy matching the Traction framework by Gino Whitman. "
        "The executive agent builds role cards and reports directly to the board of directors. "
        "Upload the repository to align repository code structures with company tracking cards."
    )
    
    compiler.ingest_dense_text(dense_data)
    compiler.compile_2d_landscape()
    
    # Save the binary model directly onto your storage drive
    model_filename = "dense_spatial_map.bin"
    compiler.save_to_binary_file(model_filename)
    
    print("\n--- Running Local Runtime Simulation ---")
    # 2. Start the Fast Runtime Router using minimal memory slots
    router = FastRuntimeRouter(model_filename)
    
    # Incoming instruction phrase
    user_prompt = "Assign roles inside the tracking agent swarm."
    print(f"Incoming Prompt: '{user_prompt}'")
    
    # Calculate target spatial sector coordinate
    target_x, target_y = router.route_incoming_prompt(user_prompt)
    print(f"Fast Routed to Spatial Sector Coordinates: X: {target_x:.2f}, Y: {target_y:.2f}")
    
    # Stream weight values strictly inside that geographical target zone
    local_weights = router.extract_nearest_weights(target_x, target_y, radius=60.0)
    
    print(f"\nSuccessfully activated {len(local_weights)} localized text nodes from Flash Drive:")
    for active_word in local_weights.keys():
        print(f" -> Active Node Loaded:
