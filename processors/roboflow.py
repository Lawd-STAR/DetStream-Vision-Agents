"""
YOLO-based Computer Vision Pre-processor for Dota 2

This module provides computer vision analysis of Dota 2 gameplay using
YOLO models via Ultralytics, similar to the workout assistant example.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import time

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class YOLOProcessor:
    """
    YOLO-based computer vision pre-processor for Dota 2 game analysis.

    This class processes video frames to extract game state information
    using YOLO models for object detection and analysis.

    Example usage:
        yolo_processor = YOLOProcessor(
            model_path="dota2_yolo.pt",
            confidence=0.6,
            device="cpu"
        )

        result = yolo_processor.process(image_data)
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",  # Default to general YOLO model
        confidence: float = 0.6,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the YOLO-based pre-processor.

        Args:
            model_path: Path to YOLO model file (.pt)
            confidence: Confidence threshold for detections (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            classes: List of class IDs to detect (None for all classes)
            **kwargs: Additional YOLO configuration options
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.classes = classes
        self.kwargs = kwargs

        self.logger = logging.getLogger("YOLOProcessor")

        # Initialize YOLO model
        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.logger.info(f"✅ Loaded YOLO model: {model_path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to load YOLO model {model_path}: {e}")
                self.model = None
        else:
            self.logger.warning("⚠️ Ultralytics not available, using mock detection")
            self.model = None

        # Dota 2 specific class mappings (customize based on your trained model)
        self.dota_classes = {
            0: "hero",
            1: "creep",
            2: "tower",
            3: "building",
            4: "neutral",
            5: "item",
            6: "ward",
            7: "courier",
        }

        # Hero position tracking for team fight detection
        self.hero_positions_history: List[List[Tuple[float, float]]] = []
        self.max_history = 10

        self.logger.info(
            f"Initialized YOLO processor: {model_path} (conf={confidence})"
        )

    def process(self, data: Any) -> Dict[str, Any]:
        """
        Process input data through YOLO analysis.

        Args:
            data: Input data to process (typically a PIL Image)

        Returns:
            Dictionary containing detection and analysis results
        """
        try:
            if isinstance(data, Image.Image):
                return self._process_image(data)
            elif isinstance(data, np.ndarray):
                # Convert numpy array to PIL Image
                if data.shape[-1] == 3:  # RGB
                    image = Image.fromarray(data.astype("uint8"), "RGB")
                else:  # Grayscale or other
                    image = Image.fromarray(data.astype("uint8"))
                return self._process_image(image)
            elif isinstance(data, dict) and "image" in data:
                return self._process_image(data["image"])
            else:
                return self._process_generic(data)

        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            return {"error": str(e), "detections": []}

    def _process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process a PIL Image through YOLO analysis."""
        self.logger.debug(f"Processing image: {image.size}")

        timestamp = time.time()

        if self.model is not None:
            # Real YOLO inference
            try:
                # Convert PIL to numpy array for YOLO
                img_array = np.array(image)

                # Run YOLO inference
                results = self.model(
                    img_array,
                    conf=self.confidence,
                    classes=self.classes,
                    device=self.device,
                    verbose=False,
                )

                # Process YOLO results
                detections = self._process_yolo_results(results[0], image.size)

            except Exception as e:
                self.logger.error(f"YOLO inference error: {e}")
                detections = self._generate_mock_detections(image.size)
        else:
            # Fallback to mock detections
            detections = self._generate_mock_detections(image.size)

        # Analyze game state from detections
        game_state = self._analyze_game_state(detections, image.size)

        # Update position history for temporal analysis
        self._update_position_history(detections)

        return {
            "detections": detections,
            "game_state": game_state,
            "image_size": image.size,
            "timestamp": timestamp,
            "analysis": {
                "hero_count": len([d for d in detections if d["class"] == "hero"]),
                "team_fight_detected": self._detect_team_fight(detections),
                "farming_opportunity": self._detect_farming_opportunity(detections),
                "danger_level": self._assess_danger_level(detections),
                "minimap_activity": self._analyze_minimap_activity(
                    detections, image.size
                ),
                "objective_focus": self._detect_objective_focus(detections),
            },
        }

    def _process_yolo_results(
        self, results, image_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Convert YOLO results to standardized detection format."""
        detections = []

        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()

            for i, box in enumerate(boxes):
                # Extract box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # Convert to center + width/height format
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                # Map class ID to Dota 2 object type
                object_class = self.dota_classes.get(class_id, f"class_{class_id}")

                detection = {
                    "class": object_class,
                    "confidence": confidence,
                    "bbox": {
                        "x": float(center_x),
                        "y": float(center_y),
                        "width": float(width),
                        "height": float(height),
                    },
                    "class_id": class_id,
                }

                # Add Dota-specific attributes based on class
                if object_class == "hero":
                    detection.update(self._analyze_hero_detection(box, image_size))
                elif object_class == "tower":
                    detection.update(self._analyze_tower_detection(box, image_size))
                elif object_class == "creep":
                    detection.update(self._analyze_creep_detection(box, image_size))

                detections.append(detection)

        return detections

    def _generate_mock_detections(
        self, image_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Generate mock detections when YOLO is not available."""
        import random

        width, height = image_size
        mock_detections = []

        # Generate some random hero detections
        for i in range(random.randint(2, 6)):
            mock_detections.append(
                {
                    "class": "hero",
                    "confidence": random.uniform(0.7, 0.95),
                    "bbox": {
                        "x": random.uniform(0.1 * width, 0.9 * width),
                        "y": random.uniform(0.1 * height, 0.9 * height),
                        "width": random.uniform(40, 80),
                        "height": random.uniform(60, 100),
                    },
                    "hero_name": random.choice(
                        ["pudge", "invoker", "crystal_maiden", "axe", "drow_ranger"]
                    ),
                    "team": random.choice(["radiant", "dire"]),
                    "health_percentage": random.uniform(0.2, 1.0),
                }
            )

        # Generate creep detections
        for i in range(random.randint(4, 12)):
            mock_detections.append(
                {
                    "class": "creep",
                    "confidence": random.uniform(0.6, 0.9),
                    "bbox": {
                        "x": random.uniform(0.2 * width, 0.8 * width),
                        "y": random.uniform(0.2 * height, 0.8 * height),
                        "width": random.uniform(25, 40),
                        "height": random.uniform(25, 40),
                    },
                    "creep_type": random.choice(["melee", "ranged", "siege"]),
                    "team": random.choice(["radiant", "dire", "neutral"]),
                }
            )

        # Generate tower detections
        for i in range(random.randint(1, 3)):
            mock_detections.append(
                {
                    "class": "tower",
                    "confidence": random.uniform(0.8, 0.98),
                    "bbox": {
                        "x": random.uniform(0.1 * width, 0.9 * width),
                        "y": random.uniform(0.1 * height, 0.9 * height),
                        "width": random.uniform(45, 60),
                        "height": random.uniform(80, 120),
                    },
                    "tower_tier": random.randint(1, 4),
                    "team": random.choice(["radiant", "dire"]),
                    "health_percentage": random.uniform(0.3, 1.0),
                }
            )

        return mock_detections

    def _analyze_hero_detection(
        self, box, image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Analyze hero-specific attributes."""
        # Mock hero analysis - in a real implementation, this could use
        # additional models or heuristics to determine hero type, team, health, etc.
        import random

        heroes = [
            "pudge",
            "invoker",
            "crystal_maiden",
            "axe",
            "drow_ranger",
            "phantom_assassin",
            "anti_mage",
            "sniper",
            "zeus",
            "shadow_fiend",
        ]

        return {
            "hero_name": random.choice(heroes),
            "team": "radiant"
            if box.xyxy[0][0] < image_size[0] / 2
            else "dire",  # Simple heuristic
            "health_percentage": random.uniform(0.2, 1.0),
            "is_visible": True,
            "action_state": random.choice(["moving", "attacking", "casting", "idle"]),
        }

    def _analyze_tower_detection(
        self, box, image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Analyze tower-specific attributes."""
        import random

        return {
            "tower_tier": random.randint(1, 4),
            "team": "radiant" if box.xyxy[0][1] > image_size[1] / 2 else "dire",
            "health_percentage": random.uniform(0.3, 1.0),
            "is_attacking": random.choice([True, False]),
        }

    def _analyze_creep_detection(
        self, box, image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Analyze creep-specific attributes."""
        import random

        return {
            "creep_type": random.choice(["melee", "ranged", "siege", "super"]),
            "team": random.choice(["radiant", "dire", "neutral"]),
            "is_last_hittable": random.choice([True, False]),
            "health_percentage": random.uniform(0.1, 1.0),
        }

    def _update_position_history(self, detections: List[Dict]) -> None:
        """Update position history for temporal analysis."""
        current_positions = []
        for detection in detections:
            if detection["class"] == "hero":
                bbox = detection["bbox"]
                current_positions.append((bbox["x"], bbox["y"]))

        self.hero_positions_history.append(current_positions)

        # Keep only recent history
        if len(self.hero_positions_history) > self.max_history:
            self.hero_positions_history.pop(0)

    def _detect_team_fight(self, detections: List[Dict]) -> bool:
        """Detect if a team fight is happening using spatial and temporal analysis."""
        heroes = [d for d in detections if d["class"] == "hero"]

        if len(heroes) < 4:
            return False

        # Spatial clustering check
        positions = [(h["bbox"]["x"], h["bbox"]["y"]) for h in heroes]
        cluster_threshold = 200  # pixels

        clustered_heroes = 0
        for i, pos1 in enumerate(positions):
            nearby_count = 0
            for j, pos2 in enumerate(positions):
                if i != j:
                    distance = (
                        (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
                    ) ** 0.5
                    if distance < cluster_threshold:
                        nearby_count += 1
            if nearby_count >= 2:
                clustered_heroes += 1

        # Team fight if multiple heroes are clustered
        return clustered_heroes >= 3

    def _detect_farming_opportunity(self, detections: List[Dict]) -> bool:
        """Detect if there's a good farming opportunity."""
        creeps = [d for d in detections if d["class"] == "creep"]
        enemy_heroes = [
            d for d in detections if d["class"] == "hero" and d.get("team") == "dire"
        ]

        # Good farming: many creeps, few enemy heroes
        return len(creeps) >= 4 and len(enemy_heroes) <= 1

    def _assess_danger_level(self, detections: List[Dict]) -> str:
        """Assess the danger level based on enemy presence and positioning."""
        enemy_heroes = [
            d for d in detections if d["class"] == "hero" and d.get("team") == "dire"
        ]
        towers = [d for d in detections if d["class"] == "tower"]

        enemy_count = len(enemy_heroes)
        safe_towers = len([t for t in towers if t.get("team") == "radiant"])

        if enemy_count >= 3:
            return "high"
        elif enemy_count >= 2 or safe_towers == 0:
            return "medium"
        else:
            return "low"

    def _analyze_minimap_activity(
        self, detections: List[Dict], image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Analyze activity in minimap region (bottom-right typically)."""
        width, height = image_size
        minimap_region = {
            "x_min": width * 0.8,
            "y_min": height * 0.8,
            "x_max": width,
            "y_max": height,
        }

        minimap_objects = []
        for detection in detections:
            bbox = detection["bbox"]
            if (
                minimap_region["x_min"] <= bbox["x"] <= minimap_region["x_max"]
                and minimap_region["y_min"] <= bbox["y"] <= minimap_region["y_max"]
            ):
                minimap_objects.append(detection)

        return {
            "objects_in_minimap": len(minimap_objects),
            "enemy_activity": len(
                [o for o in minimap_objects if o.get("team") == "dire"]
            ),
            "high_activity": len(minimap_objects) > 5,
        }

    def _detect_objective_focus(self, detections: List[Dict]) -> Dict[str, Any]:
        """Detect if players are focusing on objectives (towers, roshan, etc.)."""
        towers = [d for d in detections if d["class"] == "tower"]
        heroes = [d for d in detections if d["class"] == "hero"]

        tower_focus = False
        if towers and heroes:
            # Check if heroes are near towers
            for tower in towers:
                nearby_heroes = 0
                for hero in heroes:
                    distance = (
                        (tower["bbox"]["x"] - hero["bbox"]["x"]) ** 2
                        + (tower["bbox"]["y"] - hero["bbox"]["y"]) ** 2
                    ) ** 0.5
                    if distance < 150:  # pixels
                        nearby_heroes += 1

                if nearby_heroes >= 2:
                    tower_focus = True
                    break

        return {
            "tower_focus": tower_focus,
            "roshan_attempt": False,  # Would need specific roshan detection
            "objective_contested": tower_focus,
        }

    def _analyze_game_state(
        self, detections: List[Dict], image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Analyze overall game state from detections."""
        heroes = [d for d in detections if d["class"] == "hero"]
        creeps = [d for d in detections if d["class"] == "creep"]
        towers = [d for d in detections if d["class"] == "tower"]

        radiant_heroes = [h for h in heroes if h.get("team") == "radiant"]
        dire_heroes = [h for h in heroes if h.get("team") == "dire"]

        return {
            "phase": self._determine_game_phase(len(heroes), len(towers)),
            "team_balance": {
                "radiant_heroes": len(radiant_heroes),
                "dire_heroes": len(dire_heroes),
                "balance_ratio": len(radiant_heroes) / max(1, len(dire_heroes)),
            },
            "objectives": {
                "towers_visible": len(towers),
                "creep_waves": len(creeps) // 4,
                "contested_areas": self._count_contested_areas(detections),
            },
            "activity_level": "high"
            if len(heroes) > 4
            else "medium"
            if len(heroes) > 2
            else "low",
        }

    def _count_contested_areas(self, detections: List[Dict]) -> int:
        """Count areas where both teams have presence."""
        radiant_positions = [
            (d["bbox"]["x"], d["bbox"]["y"])
            for d in detections
            if d["class"] == "hero" and d.get("team") == "radiant"
        ]
        dire_positions = [
            (d["bbox"]["x"], d["bbox"]["y"])
            for d in detections
            if d["class"] == "hero" and d.get("team") == "dire"
        ]

        contested_count = 0
        threshold = 300  # pixels

        for r_pos in radiant_positions:
            for d_pos in dire_positions:
                distance = (
                    (r_pos[0] - d_pos[0]) ** 2 + (r_pos[1] - d_pos[1]) ** 2
                ) ** 0.5
                if distance < threshold:
                    contested_count += 1
                    break  # Count each radiant hero only once

        return contested_count

    def _determine_game_phase(self, hero_count: int, tower_count: int) -> str:
        """Determine the current phase of the game."""
        if tower_count >= 6:
            return "early_game"
        elif tower_count >= 3:
            return "mid_game"
        else:
            return "late_game"

    def _process_generic(self, data: Any) -> Dict[str, Any]:
        """Process generic data that's not an image."""
        self.logger.debug(f"Processing generic data: {type(data)}")

        return {
            "processed_data": str(data),
            "data_type": str(type(data)),
            "timestamp": time.time(),
            "analysis": {
                "data_processed": True,
                "has_useful_info": len(str(data)) > 10,
            },
        }

    def __repr__(self) -> str:
        """String representation of the processor."""
        return f"YOLOProcessor(model='{self.model_path}', conf={self.confidence})"
