"""
Comprehensive household item taxonomy for YOLO-World detection.
Organized by room type and category for efficient prompt generation.
"""

class HouseholdTaxonomy:
    """Manages household item categories for object detection."""

    # Core furniture items (applies to multiple rooms)
    FURNITURE = [
        'chair', 'armchair', 'recliner', 'office chair', 'dining chair',
        'table', 'coffee table', 'side table', 'end table', 'dining table', 'desk',
        'sofa', 'couch', 'loveseat', 'sectional', 'ottoman',
        'bed', 'bunk bed', 'crib', 'mattress',
        'dresser', 'chest of drawers', 'wardrobe', 'armoire', 'closet',
        'nightstand', 'bedside table',
        'bookshelf', 'bookcase', 'shelf', 'shelving unit', 'cabinet',
        'bench', 'stool', 'bar stool',
        'filing cabinet', 'storage cabinet'
    ]

    # Lighting fixtures
    LIGHTING = [
        'lamp', 'table lamp', 'desk lamp', 'floor lamp', 'reading lamp',
        'chandelier', 'pendant light', 'ceiling light', 'ceiling fan with light',
        'wall sconce', 'wall lamp', 'track lighting',
        'string lights', 'LED strip', 'night light'
    ]

    # Electronics and technology
    ELECTRONICS = [
        'television', 'TV', 'monitor', 'display screen',
        'computer', 'desktop computer', 'laptop', 'notebook',
        'tablet', 'iPad', 'tablet computer',
        'smartphone', 'cell phone', 'mobile phone',
        'keyboard', 'mouse', 'trackpad',
        'printer', 'scanner', 'copier',
        'router', 'modem', 'network switch',
        'speaker', 'bluetooth speaker', 'soundbar', 'home theater system',
        'game console', 'gaming system', 'PlayStation', 'Xbox', 'Nintendo',
        'camera', 'webcam', 'security camera',
        'projector', 'smart home hub', 'voice assistant',
        'charging station', 'power strip', 'surge protector',
        'headphones', 'earbuds', 'headset'
    ]

    # Kitchen appliances and items
    KITCHEN_LARGE_APPLIANCES = [
        'refrigerator', 'fridge', 'freezer',
        'oven', 'stove', 'range', 'cooktop',
        'microwave', 'microwave oven',
        'dishwasher', 'garbage disposal', 'trash compactor'
    ]

    KITCHEN_SMALL_APPLIANCES = [
        'coffee maker', 'coffee machine', 'espresso machine',
        'toaster', 'toaster oven',
        'blender', 'food processor', 'mixer', 'stand mixer', 'hand mixer',
        'air fryer', 'instant pot', 'pressure cooker', 'slow cooker', 'rice cooker',
        'kettle', 'electric kettle', 'tea kettle',
        'juicer', 'smoothie maker',
        'can opener', 'electric can opener',
        'bread maker', 'waffle maker', 'panini press', 'sandwich maker',
        'ice cream maker', 'popcorn maker'
    ]

    KITCHEN_ITEMS = [
        'sink', 'faucet', 'tap',
        'cutting board', 'knife block', 'knife set',
        'pot', 'pan', 'skillet', 'wok', 'dutch oven',
        'baking sheet', 'baking pan', 'muffin tin',
        'mixing bowl', 'bowl', 'serving bowl',
        'plate', 'dish', 'dinner plate', 'salad plate',
        'cup', 'mug', 'glass', 'wine glass', 'tumbler',
        'utensil holder', 'utensils', 'silverware', 'cutlery',
        'dish rack', 'drying rack', 'dish drainer',
        'paper towel holder', 'soap dispenser', 'sponge',
        'trash can', 'garbage can', 'recycling bin',
        'spice rack', 'spice jar', 'container set',
        'kitchen scale', 'measuring cups', 'measuring spoons',
        'oven mitt', 'pot holder', 'apron',
        'dish soap', 'cleaning supplies', 'spray bottle'
    ]

    # Decor and accessories
    DECOR = [
        'painting', 'picture', 'artwork', 'wall art', 'canvas',
        'picture frame', 'photo frame', 'photo display',
        'mirror', 'wall mirror', 'floor mirror', 'vanity mirror',
        'rug', 'area rug', 'carpet', 'floor mat',
        'curtain', 'drapes', 'blinds', 'window shade', 'window treatment',
        'pillow', 'throw pillow', 'decorative pillow', 'cushion',
        'throw blanket', 'blanket', 'afghan',
        'vase', 'flower vase', 'decorative vase',
        'plant', 'potted plant', 'indoor plant', 'houseplant', 'succulent',
        'planter', 'flower pot', 'plant stand',
        'sculpture', 'figurine', 'statue', 'decorative object',
        'candle', 'candle holder', 'candlestick',
        'clock', 'wall clock', 'desk clock', 'alarm clock',
        'tapestry', 'wall hanging', 'macrame'
    ]

    # Bedroom items
    BEDROOM = [
        'bed', 'mattress', 'box spring', 'bed frame',
        'pillow', 'bed pillow', 'throw pillow',
        'comforter', 'duvet', 'bedspread', 'quilt',
        'sheet', 'fitted sheet', 'flat sheet', 'pillowcase',
        'blanket', 'throw blanket', 'afghan',
        'nightstand', 'bedside table',
        'dresser', 'chest of drawers',
        'wardrobe', 'armoire', 'closet organizer',
        'laundry basket', 'hamper',
        'jewelry box', 'jewelry organizer',
        'alarm clock', 'clock radio'
    ]

    # Bathroom items
    BATHROOM = [
        'toilet', 'toilet seat', 'toilet paper holder',
        'sink', 'bathroom sink', 'vanity sink',
        'bathtub', 'tub', 'shower', 'shower stall',
        'shower curtain', 'shower curtain rod',
        'towel', 'bath towel', 'hand towel', 'washcloth',
        'towel rack', 'towel bar', 'towel ring', 'towel hook',
        'bath mat', 'shower mat', 'toilet mat',
        'mirror', 'bathroom mirror', 'medicine cabinet',
        'scale', 'bathroom scale', 'weight scale',
        'soap dispenser', 'soap dish', 'toothbrush holder',
        'toilet brush', 'plunger', 'toilet paper',
        'shampoo', 'conditioner', 'body wash', 'soap',
        'hair dryer', 'curling iron', 'straightener',
        'razor', 'electric toothbrush'
    ]

    # Office items
    OFFICE = [
        'desk', 'writing desk', 'computer desk', 'standing desk',
        'office chair', 'desk chair', 'ergonomic chair',
        'filing cabinet', 'file organizer', 'drawer organizer',
        'bookshelf', 'bookcase',
        'desk lamp', 'task lamp',
        'computer', 'monitor', 'keyboard', 'mouse',
        'printer', 'scanner', 'shredder',
        'whiteboard', 'bulletin board', 'corkboard',
        'desk organizer', 'pen holder', 'pencil holder',
        'stapler', 'tape dispenser', 'scissors',
        'notebook', 'binder', 'folder',
        'calendar', 'planner', 'notepad'
    ]

    # Storage and organization
    STORAGE = [
        'box', 'storage box', 'plastic bin', 'storage bin',
        'basket', 'wicker basket', 'storage basket',
        'container', 'storage container', 'organizer',
        'shelf', 'shelving unit', 'storage shelf',
        'cabinet', 'storage cabinet',
        'chest', 'trunk', 'storage trunk',
        'bag', 'tote bag', 'shopping bag',
        'suitcase', 'luggage', 'travel bag',
        'backpack', 'duffel bag', 'gym bag'
    ]

    # Cleaning and maintenance
    CLEANING = [
        'vacuum', 'vacuum cleaner', 'robot vacuum',
        'broom', 'mop', 'bucket',
        'dustpan', 'cleaning cloth', 'microfiber cloth',
        'iron', 'ironing board',
        'drying rack', 'clothes drying rack',
        'laundry detergent', 'fabric softener',
        'cleaning spray', 'cleaning supplies'
    ]

    # Climate control
    CLIMATE = [
        'fan', 'ceiling fan', 'floor fan', 'desk fan', 'tower fan',
        'air conditioner', 'AC unit', 'portable AC',
        'heater', 'space heater', 'electric heater',
        'humidifier', 'dehumidifier',
        'air purifier', 'air filter',
        'thermostat', 'smart thermostat'
    ]

    # Entertainment and media
    ENTERTAINMENT = [
        'television', 'TV', 'TV stand', 'entertainment center',
        'speaker', 'bluetooth speaker', 'soundbar',
        'record player', 'turntable', 'vinyl player',
        'DVD player', 'Blu-ray player',
        'game console', 'gaming system',
        'remote control', 'universal remote',
        'media player', 'streaming device', 'Roku', 'Apple TV', 'Fire TV',
        'book', 'magazine', 'newspaper',
        'board game', 'puzzle', 'toy'
    ]

    # Children's items
    CHILDREN = [
        'crib', 'bassinet', 'changing table',
        'high chair', 'booster seat', 'baby chair',
        'toy', 'toy box', 'toy organizer',
        'stuffed animal', 'teddy bear', 'doll',
        'baby monitor', 'baby gate',
        'diaper pail', 'diaper bag',
        'stroller', 'car seat',
        'play mat', 'activity mat'
    ]

    @classmethod
    def get_all_items(cls):
        """Get all household items as a flat list."""
        all_items = []
        all_items.extend(cls.FURNITURE)
        all_items.extend(cls.LIGHTING)
        all_items.extend(cls.ELECTRONICS)
        all_items.extend(cls.KITCHEN_LARGE_APPLIANCES)
        all_items.extend(cls.KITCHEN_SMALL_APPLIANCES)
        all_items.extend(cls.KITCHEN_ITEMS)
        all_items.extend(cls.DECOR)
        all_items.extend(cls.BEDROOM)
        all_items.extend(cls.BATHROOM)
        all_items.extend(cls.OFFICE)
        all_items.extend(cls.STORAGE)
        all_items.extend(cls.CLEANING)
        all_items.extend(cls.CLIMATE)
        all_items.extend(cls.ENTERTAINMENT)
        all_items.extend(cls.CHILDREN)

        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in all_items:
            if item.lower() not in seen:
                seen.add(item.lower())
                unique_items.append(item)

        return unique_items

    @classmethod
    def get_room_specific_items(cls, room_type):
        """Get items specific to a room type."""
        room_mappings = {
            'kitchen': cls.KITCHEN_LARGE_APPLIANCES + cls.KITCHEN_SMALL_APPLIANCES + cls.KITCHEN_ITEMS,
            'bedroom': cls.BEDROOM + cls.FURNITURE + cls.LIGHTING + cls.DECOR,
            'bathroom': cls.BATHROOM + cls.CLEANING,
            'living_room': cls.FURNITURE + cls.ENTERTAINMENT + cls.LIGHTING + cls.DECOR,
            'office': cls.OFFICE + cls.FURNITURE + cls.ELECTRONICS + cls.LIGHTING,
            'nursery': cls.CHILDREN + cls.BEDROOM + cls.FURNITURE
        }

        return room_mappings.get(room_type.lower(), cls.get_all_items())

    @classmethod
    def get_item_count(cls):
        """Get total number of unique items in taxonomy."""
        return len(cls.get_all_items())


# Example usage:
if __name__ == "__main__":
    taxonomy = HouseholdTaxonomy()

    print(f"Total household items: {taxonomy.get_item_count()}")
    print(f"\nFirst 20 items: {taxonomy.get_all_items()[:20]}")
    print(f"\nKitchen items: {len(taxonomy.get_room_specific_items('kitchen'))}")
    print(f"Bedroom items: {len(taxonomy.get_room_specific_items('bedroom'))}")
