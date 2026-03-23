"""
PikoLab — Base de conseils holistiques pour les 16 saisons.
Chaque saison contient : description, palettes catégorisées, maquillage,
vêtements, cheveux, accessoires.
Données basées sur la théorie des couleurs saisonnières (Munsell / Sci\ART).
"""

SEASON_ADVICE = {
    # ================================================================
    # SPRING — Chaud + Clair
    # ================================================================
    "Light Spring": {
        "description": (
            "Douceur lumineuse. Couleurs claires, chaudes et delicates comme un matin de printemps. "
            "Votre palette est la plus legere des saisons chaudes : pastel chauds, tons peche, ivoire."
        ),
        "palette_neutrals": ["#F5F0E1", "#D2B48C", "#C4A882", "#8B7D6B", "#5C5248"],
        "palette_accents": ["#FFB347", "#FF6F61", "#87CEAB", "#FFD700", "#F4A460", "#77B5FE"],
        "palette_avoid": ["#000000", "#1C1C1C", "#800020", "#4B0082", "#2F4F4F", "#C0C0C0"],
        "metals": "Or jaune clair, or rose. Eviter argent et platine.",
        "makeup": {
            "foundation": "Sous-ton peche/dore clair. Eviter les fonds de teint roses ou trop couvrants.",
            "lips": ["Peche clair", "Corail doux", "Rose saumon", "Nude abricot"],
            "eyes": ["Champagne", "Peche", "Vert sauge clair", "Brun dore clair"],
            "blush": ["Peche", "Abricot clair", "Rose corail lumineux"],
            "eyebrows": "Blond dore ou chatain clair, trait leger"
        },
        "clothing": {
            "best_combinations": [
                "Ivoire + peche",
                "Camel clair + corail",
                "Bleu ciel chaud + creme",
                "Vert d'eau + dore"
            ],
            "patterns": "Motifs floraux delicats, rayures fines, aquarelle, liberty",
            "contrast_tip": "Contraste bas a moyen. Harmonies ton sur ton. Eviter les forts contrastes noir/blanc.",
            "fabrics": "Coton leger, lin, soie fluide, mousseline, maille fine"
        },
        "hair": {
            "ideal": ["Blond dore", "Blond miel", "Chatain clair dore", "Blond venitien"],
            "avoid": ["Noir", "Brun froid", "Blond platine", "Cendre"]
        },
        "accessories": {
            "glasses": "Montures dorees, transparentes, nude, ecaille claire",
            "jewelry": "Or jaune, perles creme, pierres pastel (quartz rose, citrine, aigue-marine)",
            "bags_shoes": "Nude, camel clair, peche, cognac clair. Pas de noir."
        }
    },

    "Warm Spring": {
        "description": (
            "Chaleur rayonnante. Couleurs riches et chaudes avec une luminosite naturelle. "
            "Votre palette est intensement doree : peche, corail, turquoise chaud, vert pomme."
        ),
        "palette_neutrals": ["#FFFDD0", "#D2B48C", "#8B7355", "#5C4033", "#3B2F2F"],
        "palette_accents": ["#E07A5F", "#E9C46A", "#81B29A", "#F4A261", "#FF6B35", "#2A9D8F"],
        "palette_avoid": ["#000000", "#FF00FF", "#C0C0C0", "#4169E1", "#800080", "#F8F8FF"],
        "metals": "Or jaune, cuivre, laiton. Eviter argent froid.",
        "makeup": {
            "foundation": "Sous-ton dore franc. Les teintes peche/abricot sont ideales.",
            "lips": ["Corail chaud", "Orange brulee doux", "Peche intense", "Brique claire"],
            "eyes": ["Bronze", "Cuivre", "Vert olive", "Brun dore", "Turquoise"],
            "blush": ["Peche intense", "Abricot", "Corail chaud"],
            "eyebrows": "Chatain dore, auburn clair"
        },
        "clothing": {
            "best_combinations": [
                "Turquoise + corail",
                "Vert olive + moutarde",
                "Rouille claire + creme",
                "Camel + vert pomme"
            ],
            "patterns": "Imprime tropical, ethnique, pois, rayures colorees",
            "contrast_tip": "Contraste moyen. Les couleurs vives sont vos alliees mais restez dans les tons chauds.",
            "fabrics": "Lin, coton, soie mate, daim leger, raphia"
        },
        "hair": {
            "ideal": ["Roux dore", "Auburn", "Chatain dore", "Blond miel fonce"],
            "avoid": ["Noir bleu", "Cendre", "Gris argente", "Blond platine"]
        },
        "accessories": {
            "glasses": "Ecaille chaude, dore, caramel, vert olive",
            "jewelry": "Or jaune, ambre, corail, turquoise, bois",
            "bags_shoes": "Cognac, tan, camel, rouille. Eviter noir et gris."
        }
    },

    "Bright Spring": {
        "description": (
            "Eclat et vivacite. Couleurs vives, pures et lumineuses. "
            "Votre palette est la plus saturee des printemps : corail vif, turquoise eclatant, jaune soleil."
        ),
        "palette_neutrals": ["#FFFAFA", "#F5DEB3", "#8B7355", "#4A4A4A", "#1C1C1C"],
        "palette_accents": ["#FF6B35", "#06D6A0", "#EF476F", "#118AB2", "#FFD166", "#F77F00"],
        "palette_avoid": ["#8B8C7A", "#B5838D", "#C2B280", "#A39BA8", "#6D6875", "#5F7A61"],
        "metals": "Or jaune brillant, or rose. L'argent est possible si brillant.",
        "makeup": {
            "foundation": "Sous-ton peche/dore lumineux. Finish satin ou lumineux.",
            "lips": ["Corail vif", "Rose chaud vif", "Rouge tomate", "Fuchsia chaud"],
            "eyes": ["Turquoise", "Vert emeraude chaud", "Bronze brillant", "Violet chaud"],
            "blush": ["Corail lumineux", "Peche vif", "Rose chaud"],
            "eyebrows": "Chatain chaud, brun dore — jamais gris"
        },
        "clothing": {
            "best_combinations": [
                "Corail vif + turquoise",
                "Jaune soleil + bleu cobalt",
                "Vert pomme + rose vif",
                "Blanc + orange eclatant"
            ],
            "patterns": "Geometriques, color-block, imprime graphique, rayures contrastees",
            "contrast_tip": "Contraste eleve. Vous portez les couleurs pures et vives. Evitez le terne.",
            "fabrics": "Soie brillante, coton sature, cuir lisse, materiaux a reflets"
        },
        "hair": {
            "ideal": ["Chatain dore vif", "Auburn eclatant", "Blond dore chaud"],
            "avoid": ["Cendre", "Gris", "Brun terne", "Noir mat"]
        },
        "accessories": {
            "glasses": "Montures colorees vives, ecaille, or brillant",
            "jewelry": "Or jaune, pierres vives (emeraude, rubis, turquoise, citrine)",
            "bags_shoes": "Couleurs vives ou cognac. Le noir est OK si brillant."
        }
    },

    "True Spring": {
        "description": (
            "L'essence du printemps. Equilibre parfait entre chaleur, clarte et vivacite. "
            "Couleurs pures et joyeuses : vert gazon, rouge coquelicot, bleu ciel."
        ),
        "palette_neutrals": ["#FFFFF0", "#D2B48C", "#8B7355", "#556B2F", "#3B3B3B"],
        "palette_accents": ["#FF5733", "#FFC300", "#28B463", "#FF69B4", "#00B4D8", "#FF8C00"],
        "palette_avoid": ["#1C1C1C", "#800020", "#4B0082", "#2F4F4F", "#808080", "#C0C0C0"],
        "metals": "Or jaune, or rose, laiton poli.",
        "makeup": {
            "foundation": "Sous-ton dore equilibre. Ni trop rose, ni trop jaune.",
            "lips": ["Corail pur", "Rouge tomate", "Rose peche", "Nude dore"],
            "eyes": ["Vert gazon", "Bronze", "Peche", "Bleu chaud", "Brun chaud"],
            "blush": ["Corail", "Peche", "Rose chaud"],
            "eyebrows": "Chatain dore moyen"
        },
        "clothing": {
            "best_combinations": [
                "Bleu cobalt + corail",
                "Vert prairie + jaune",
                "Ivoire + rouge coquelicot",
                "Turquoise + peche"
            ],
            "patterns": "Floraux, rayures, pois, motifs naturels, mix & match",
            "contrast_tip": "Contraste moyen a eleve. Couleurs pures, jamais trop sombres ou trop passees.",
            "fabrics": "Coton, lin colore, soie, jersey, denim"
        },
        "hair": {
            "ideal": ["Chatain dore", "Blond chaud moyen", "Auburn", "Roux naturel"],
            "avoid": ["Noir profond", "Cendre", "Gris", "Blond platine"]
        },
        "accessories": {
            "glasses": "Dore, ecaille, transparent, couleurs vives",
            "jewelry": "Or jaune, pierres colorees vives, email, bijoux fantaisie colores",
            "bags_shoes": "Cognac, tan, couleurs vives, blanc casse. Noir a eviter."
        }
    },

    # ================================================================
    # SUMMER — Froid + Clair
    # ================================================================
    "Light Summer": {
        "description": (
            "Douceur etheree. Couleurs froides, claires et delicates comme une brume estivale. "
            "Pastels froids, lavande, bleu poudre, rose doux."
        ),
        "palette_neutrals": ["#F0F0F0", "#C0C0C0", "#A9A9B0", "#6B6B7B", "#4A4A5A"],
        "palette_accents": ["#CDB4DB", "#BDE0FE", "#FFAFCC", "#A2D2FF", "#D0D1FF", "#98D8C8"],
        "palette_avoid": ["#FF4500", "#FF8C00", "#8B4513", "#DAA520", "#556B2F", "#000000"],
        "metals": "Argent, or blanc, platine, or rose clair.",
        "makeup": {
            "foundation": "Sous-ton rose clair/neutre froid. Finish naturel.",
            "lips": ["Rose poudre", "Mauve clair", "Rose nude froid", "Framboise douce"],
            "eyes": ["Lavande", "Gris perle", "Rose froid", "Bleu doux", "Taupe froid"],
            "blush": ["Rose doux", "Mauve clair", "Rose frais"],
            "eyebrows": "Blond cendre, taupe froid, gris doux"
        },
        "clothing": {
            "best_combinations": [
                "Lavande + gris perle",
                "Rose poudre + bleu ciel",
                "Creme froid + mauve",
                "Bleu poudre + blanc doux"
            ],
            "patterns": "Floraux delicats, aquarelle, rayures fines, ton sur ton",
            "contrast_tip": "Contraste bas. Tout en douceur, pas de rupture brutale.",
            "fabrics": "Mousseline, soie fluide, coton doux, cachemire fin, organza"
        },
        "hair": {
            "ideal": ["Blond cendre clair", "Blond platine doux", "Chatain clair cendre"],
            "avoid": ["Roux", "Auburn", "Noir", "Chatain dore"]
        },
        "accessories": {
            "glasses": "Argent, transparent, rose pale, lilas",
            "jewelry": "Argent, or blanc, perles, quartz rose, aigue-marine, opale",
            "bags_shoes": "Gris clair, taupe froid, rose poudre, bleu gris. Pas de marron chaud."
        }
    },

    "Cool Summer": {
        "description": (
            "Elegance froide. Couleurs froides, moyennement profondes, sophistiquees. "
            "Bleu acier, rose cendre, bordeaux froid, gris bleu."
        ),
        "palette_neutrals": ["#F0EFF4", "#9A8C98", "#6D6875", "#4A4E69", "#2B2D42"],
        "palette_accents": ["#7B9EA8", "#C9ADA7", "#84A98C", "#B5838D", "#9CADB7", "#8B687D"],
        "palette_avoid": ["#FF8C00", "#FFD700", "#FF4500", "#DAA520", "#8B4513", "#556B2F"],
        "metals": "Argent, platine, or blanc.",
        "makeup": {
            "foundation": "Sous-ton rose/neutre froid. Eviter le dore.",
            "lips": ["Rose mauve", "Framboise", "Prune douce", "Rose-brun froid"],
            "eyes": ["Gris bleu", "Taupe froid", "Mauve", "Argent", "Prune"],
            "blush": ["Rose froid", "Mauve", "Framboise douce"],
            "eyebrows": "Brun froid, taupe, cendre"
        },
        "clothing": {
            "best_combinations": [
                "Gris bleu + rose cendre",
                "Marine froid + lavande",
                "Bordeaux froid + gris perle",
                "Bleu acier + blanc froid"
            ],
            "patterns": "Classiques, Prince-de-Galles, rayures, geometriques sobres",
            "contrast_tip": "Contraste moyen. Sophistication et harmonie froide.",
            "fabrics": "Laine fine, cachemire, soie mate, crepe, gabardine"
        },
        "hair": {
            "ideal": ["Chatain cendre", "Brun froid", "Blond fonce cendre"],
            "avoid": ["Roux", "Dore", "Auburn", "Caramel"]
        },
        "accessories": {
            "glasses": "Argent, gris, bleu marine, noir doux",
            "jewelry": "Argent, platine, perles grises, saphir, tanzanite, améthyste",
            "bags_shoes": "Noir, gris, marine, bordeaux froid, taupe. Pas de marron ni camel."
        }
    },

    "Soft Summer": {
        "description": (
            "Douceur feutree. Couleurs desaturees, froides et voilees. "
            "Rien de vif ni de criard. Rose poudre, sauge, bleu gris, mauve."
        ),
        "palette_neutrals": ["#EDEBE8", "#C9CBA3", "#A39BA8", "#8D99AE", "#5A5A6E"],
        "palette_accents": ["#B5838D", "#95B8D1", "#CEB5A7", "#A8A4CE", "#B8B8D1", "#8DAA9D"],
        "palette_avoid": ["#FF0000", "#FF8C00", "#00FF00", "#FFD700", "#FF1493", "#000000"],
        "metals": "Argent mat, or rose, or blanc. Eviter les metaux brillants.",
        "makeup": {
            "foundation": "Sous-ton neutre a froid. Couvrance legere, finish mat doux.",
            "lips": ["Rose poudre", "Mauve", "Nude rose froid", "Baie douce"],
            "eyes": ["Taupe", "Gris lavande", "Rose fane", "Vert sauge", "Brun doux froid"],
            "blush": ["Rose sourd", "Mauve doux", "Peche froid"],
            "eyebrows": "Taupe, brun doux froid — jamais de trait net"
        },
        "clothing": {
            "best_combinations": [
                "Sauge + rose fane",
                "Bleu gris + mauve",
                "Taupe + lavande",
                "Gris perle + vieux rose"
            ],
            "patterns": "Ton sur ton, imprime discret, paisley doux, melange, tweed pastel",
            "contrast_tip": "Contraste BAS. C'est votre marque : tout en harmonie douce, jamais de rupture.",
            "fabrics": "Cachemire, laine merinos, jersey doux, daim, lin lave, maille"
        },
        "hair": {
            "ideal": ["Chatain moyen cendre", "Blond fonce cendre", "Brun doux cendre"],
            "avoid": ["Noir profond", "Roux vif", "Blond dore", "Meches contrastees"]
        },
        "accessories": {
            "glasses": "Taupe, gris doux, transparent, rose poudre",
            "jewelry": "Argent mat, or rose, perles, quartz rose, labradorite, pierres de lune",
            "bags_shoes": "Taupe, gris, vieux rose, bleu gris. Eviter noir pur et marron chaud."
        }
    },

    "True Summer": {
        "description": (
            "L'essence de l'ete. Equilibre entre fraicheur, clarte et douceur. "
            "Rose vrai, bleu moyen, vert ocean, pervenche."
        ),
        "palette_neutrals": ["#F0F0F0", "#B0B0BC", "#6B6B7B", "#4A4E69", "#1D3557"],
        "palette_accents": ["#DB7093", "#2A6F97", "#66CDAA", "#7678ED", "#C77DFF", "#BA68C8"],
        "palette_avoid": ["#FF4500", "#DAA520", "#8B4513", "#FF8C00", "#556B2F", "#000000"],
        "metals": "Argent, or blanc, platine, or rose.",
        "makeup": {
            "foundation": "Sous-ton rose equilibre. Ni trop chaud, ni trop froid.",
            "lips": ["Rose vrai", "Framboise", "Mauve rose", "Rose the"],
            "eyes": ["Pervenche", "Gris bleu", "Rose", "Vert ocean", "Prune douce"],
            "blush": ["Rose vrai", "Framboise douce", "Rose mauve"],
            "eyebrows": "Brun moyen froid, taupe"
        },
        "clothing": {
            "best_combinations": [
                "Marine + rose",
                "Pervenche + blanc",
                "Vert ocean + gris",
                "Prune douce + bleu ciel"
            ],
            "patterns": "Rayures classiques, floraux moyens, imprime geometrique doux",
            "contrast_tip": "Contraste moyen. Couleurs fraiches mais jamais criardes.",
            "fabrics": "Coton, jersey, soie, lin, laine legere"
        },
        "hair": {
            "ideal": ["Chatain cendre moyen", "Brun moyen froid", "Blond fonce cendre"],
            "avoid": ["Roux", "Dore", "Noir bleu", "Caramel"]
        },
        "accessories": {
            "glasses": "Argent, gris bleu, marine, transparent bleu",
            "jewelry": "Argent, or blanc, perles, saphir, tanzanite, aigue-marine",
            "bags_shoes": "Marine, gris, taupe froid, bordeaux froid."
        }
    },

    # ================================================================
    # AUTUMN — Chaud + Sombre
    # ================================================================
    "Soft Autumn": {
        "description": (
            "Douceur terreuse. Couleurs chaudes, desaturees et feutrees. "
            "Camel, sauge, terre de sienne, kaki, tout en nuances sourdes."
        ),
        "palette_neutrals": ["#EDE8D5", "#C2B280", "#8B8C7A", "#5F5B50", "#3E3D36"],
        "palette_accents": ["#A0937D", "#8DAA9D", "#C9ADA7", "#7F6B5D", "#5F7A61", "#9B8E7E"],
        "palette_avoid": ["#FF0000", "#FF00FF", "#00BFFF", "#FFD700", "#000000", "#F8F8FF"],
        "metals": "Or jaune mat, laiton vieilli, or rose. Eviter argent brillant.",
        "makeup": {
            "foundation": "Sous-ton peche/neutre chaud. Couvrance legere, finish mat naturel.",
            "lips": ["Nude peche", "Rose terre", "Brique douce", "Caramel"],
            "eyes": ["Taupe chaud", "Kaki", "Brun doux", "Vert sauge", "Bronze mat"],
            "blush": ["Peche sourde", "Abricot doux", "Terre cuite claire"],
            "eyebrows": "Chatain chaud doux, taupe chaud"
        },
        "clothing": {
            "best_combinations": [
                "Camel + sauge",
                "Taupe chaud + rose terre",
                "Kaki + creme chaud",
                "Olive doux + peche fane"
            ],
            "patterns": "Paisley, tartans doux, motifs naturels, imprime animal discret, melange",
            "contrast_tip": "Contraste BAS. Harmonies ton sur ton, camaieu terreux.",
            "fabrics": "Daim, laine feutree, lin lave, cachemire, coton epais, velours cotele"
        },
        "hair": {
            "ideal": ["Chatain chaud moyen", "Blond fonce dore", "Brun chaud doux"],
            "avoid": ["Noir", "Blond platine", "Cendre", "Roux vif"]
        },
        "accessories": {
            "glasses": "Ecaille douce, taupe, nude, kaki",
            "jewelry": "Or jaune mat, bois, pierres naturelles (jaspe, agate, oeil de tigre)",
            "bags_shoes": "Camel, taupe, cognac, olive. Pas de noir ni blanc pur."
        }
    },

    "Warm Autumn": {
        "description": (
            "Richesse terreuse. Couleurs chaudes, profondes et naturelles. "
            "Rouille, moutarde, vert foret, brique — la terre dans toute sa splendeur."
        ),
        "palette_neutrals": ["#F5E6CA", "#8B7355", "#5C4033", "#2E3B2E", "#2D1B0E"],
        "palette_accents": ["#BC6C25", "#DDA15E", "#606C38", "#E76F51", "#CC5803", "#D4A017"],
        "palette_avoid": ["#FF69B4", "#00BFFF", "#C0C0C0", "#F0F0F0", "#FF00FF", "#4169E1"],
        "metals": "Or jaune, bronze, cuivre. Eviter argent froid.",
        "makeup": {
            "foundation": "Sous-ton dore/peche fonce. Eviter les fonds de teint roses.",
            "lips": ["Terracotta", "Brique", "Rouille", "Nude chaud fonce"],
            "eyes": ["Brun chaud", "Kaki", "Bronze", "Cuivre", "Olive", "Rouille"],
            "blush": ["Peche fonce", "Abricot", "Terracotta doux"],
            "eyebrows": "Brun chaud, auburn — jamais gris ni cendre"
        },
        "clothing": {
            "best_combinations": [
                "Camel + vert olive",
                "Rouille + creme chaud",
                "Brun chocolat + orange brulee",
                "Moutarde + vert foret"
            ],
            "patterns": "Paisley, tartans chauds, imprime ethnique, pied-de-poule terreux",
            "contrast_tip": "Contraste moyen. Eviter noir/blanc pur. Le marine chaud remplace le noir.",
            "fabrics": "Tweed, velours, daim, laine, cuir vieilli, velours cotele"
        },
        "hair": {
            "ideal": ["Chatain dore fonce", "Auburn", "Brun chaud", "Cuivre"],
            "avoid": ["Blond platine", "Noir bleu", "Cendre", "Gris"]
        },
        "accessories": {
            "glasses": "Ecaille chaude, bronze, havane, ambre",
            "jewelry": "Or jaune, ambre, corail, turquoise, bois, cuivre",
            "bags_shoes": "Brun, cognac, camel, olive. Pas de noir pur."
        }
    },

    "Deep Autumn": {
        "description": (
            "Profondeur chaleureuse. Les couleurs les plus sombres et riches des saisons chaudes. "
            "Bordeaux, vert sapin, brun chocolat, prune chaude — opulence terreuse."
        ),
        "palette_neutrals": ["#E8DCC8", "#7F6B5D", "#4A3728", "#2E3B2E", "#1A1A1A"],
        "palette_accents": ["#6B2737", "#8B4513", "#1B5E20", "#8B6914", "#722F37", "#4A5240"],
        "palette_avoid": ["#FFAFCC", "#BDE0FE", "#FFD700", "#98FB98", "#F0F0F0", "#C0C0C0"],
        "metals": "Or jaune profond, bronze antique, cuivre fonce.",
        "makeup": {
            "foundation": "Sous-ton dore profond/olive. Couvrance moyenne a forte.",
            "lips": ["Bordeaux chaud", "Brun rouge", "Prune chaude", "Chocolat"],
            "eyes": ["Bronze fonce", "Vert foret", "Brun intense", "Cuivre", "Olive fonce"],
            "blush": ["Terracotta", "Prune chaude", "Brique"],
            "eyebrows": "Brun fonce chaud, jamais noir pur"
        },
        "clothing": {
            "best_combinations": [
                "Bordeaux + vert sapin",
                "Chocolat + or fonce",
                "Olive fonce + rouille",
                "Prune chaude + camel"
            ],
            "patterns": "Tartans sombres, brocart, imprime baroque, motifs riches",
            "contrast_tip": "Contraste moyen a eleve. Profondeur et richesse sans le noir pur.",
            "fabrics": "Velours, soie epaisse, cuir, laine bouillie, brocart, satin mat"
        },
        "hair": {
            "ideal": ["Brun chaud profond", "Auburn fonce", "Chatain fonce dore"],
            "avoid": ["Blond clair", "Cendre", "Gris", "Noir bleu"]
        },
        "accessories": {
            "glasses": "Ecaille foncee, bronze, brun chaud",
            "jewelry": "Or jaune, grenat, oeil de tigre, onyx, citrine fumee",
            "bags_shoes": "Brun fonce, chocolat, bordeaux, olive fonce."
        }
    },

    "True Autumn": {
        "description": (
            "L'essence de l'automne. Equilibre parfait entre chaleur, profondeur et richesse. "
            "Citrouille, olive, rouille, brun — les couleurs de la recolte."
        ),
        "palette_neutrals": ["#F0E6D2", "#8B7D6B", "#5C4033", "#3B3B2E", "#2D1B0E"],
        "palette_accents": ["#D35400", "#6B8E23", "#B7410E", "#795548", "#B8860B", "#008080"],
        "palette_avoid": ["#FF69B4", "#4169E1", "#C0C0C0", "#F8F8FF", "#800080", "#00BFFF"],
        "metals": "Or jaune, bronze, cuivre, laiton.",
        "makeup": {
            "foundation": "Sous-ton dore equilibre. La peche et l'abricot sont vos bases.",
            "lips": ["Rouille", "Brique", "Orange brulee", "Nude terreux"],
            "eyes": ["Bronze", "Olive", "Brun terreux", "Cuivre", "Kaki"],
            "blush": ["Terracotta", "Abricot fonce", "Peche brulee"],
            "eyebrows": "Brun chaud moyen, auburn"
        },
        "clothing": {
            "best_combinations": [
                "Rouille + olive",
                "Citrouille + brun",
                "Teal + camel",
                "Moutarde + bordeaux chaud"
            ],
            "patterns": "Tartans, paisley, pied-de-poule, motifs automnaux",
            "contrast_tip": "Contraste moyen. Richesse naturelle sans extremes.",
            "fabrics": "Laine, tweed, velours cotele, daim, cuir, flanelle"
        },
        "hair": {
            "ideal": ["Chatain chaud", "Auburn moyen", "Brun chaud moyen", "Cuivre"],
            "avoid": ["Noir", "Blond platine", "Cendre", "Gris"]
        },
        "accessories": {
            "glasses": "Ecaille, bronze, tortoise, olive",
            "jewelry": "Or jaune, ambre, jaspe, cornaline, turquoise",
            "bags_shoes": "Cognac, brun, camel, olive, rouille."
        }
    },

    # ================================================================
    # WINTER — Froid + Sombre
    # ================================================================
    "Deep Winter": {
        "description": (
            "Intensite dramatique. Les couleurs les plus profondes et contrastees. "
            "Noir, blanc pur, rouge vif, emeraude, bleu royal — puissance et contraste."
        ),
        "palette_neutrals": ["#FFFFFF", "#C0C0C0", "#36454F", "#1C1C1C", "#000000"],
        "palette_accents": ["#CC0000", "#1A5276", "#0B6623", "#FF1493", "#6A0DAD", "#B22222"],
        "palette_avoid": ["#DDA15E", "#C2B280", "#F4A261", "#8B8C7A", "#D2B48C", "#FADADD"],
        "metals": "Argent, platine, or blanc. Noir et chrome.",
        "makeup": {
            "foundation": "Sous-ton neutre a froid, profond. Couvrance moyenne a forte.",
            "lips": ["Rouge profond", "Bordeaux froid", "Prune intense", "Baie foncee"],
            "eyes": ["Noir charbon", "Bleu nuit", "Emeraude fonce", "Argent fonce", "Prune"],
            "blush": ["Rose profond", "Berry froid", "Prune"],
            "eyebrows": "Brun fonce a noir, trait net et defini"
        },
        "clothing": {
            "best_combinations": [
                "Noir + rouge vif",
                "Marine profond + blanc",
                "Emeraude + argent",
                "Aubergine + blanc"
            ],
            "patterns": "Geometriques forts, color-block, rayures contrastees, uni",
            "contrast_tip": "Contraste ELEVE. Vous etes fait(e) pour le noir et blanc. Les contrastes forts vous subliment.",
            "fabrics": "Soie, satin, laine fine, cuir lisse, velours, coton sature"
        },
        "hair": {
            "ideal": ["Noir profond", "Brun fonce froid", "Chatain fonce espresso"],
            "avoid": ["Roux", "Dore", "Caramel", "Blond miel"]
        },
        "accessories": {
            "glasses": "Noir, argent, bleu marine, formes nettes",
            "jewelry": "Argent, platine, diamant, onyx, saphir, rubis, emeraude",
            "bags_shoes": "Noir, marine, bordeaux froid. Le blanc est OK."
        }
    },

    "Cool Winter": {
        "description": (
            "Fraicheur glacee. Couleurs froides, vives et nettes. "
            "Bleu glace, fuchsia, rouge cerise, vert pin — la clarte du givre."
        ),
        "palette_neutrals": ["#F8F8FF", "#B0B0BC", "#36454F", "#2B2D42", "#0D0D0D"],
        "palette_accents": ["#DC143C", "#4B0082", "#C71585", "#01796F", "#B0E0E6", "#191970"],
        "palette_avoid": ["#FF8C00", "#DAA520", "#8B4513", "#DDA15E", "#556B2F", "#F4A261"],
        "metals": "Argent, platine, or blanc. Chrome.",
        "makeup": {
            "foundation": "Sous-ton rose franc. Froid et net.",
            "lips": ["Rouge cerise", "Fuchsia", "Rose vif froid", "Prune froide"],
            "eyes": ["Bleu glace", "Argent", "Gris charbon", "Violet froid", "Vert pin"],
            "blush": ["Rose vif froid", "Fuchsia doux", "Berry"],
            "eyebrows": "Brun froid net, noir doux"
        },
        "clothing": {
            "best_combinations": [
                "Noir + fuchsia",
                "Marine + blanc glace",
                "Gris charbon + rouge cerise",
                "Vert pin + argent"
            ],
            "patterns": "Geometriques nets, rayures contrastees, pied-de-poule, minimaliste",
            "contrast_tip": "Contraste eleve. Nettete et precision. Couleurs pures, pas de muted.",
            "fabrics": "Soie froide, coton net, laine fine, cuir, satin, popeline"
        },
        "hair": {
            "ideal": ["Noir bleu", "Brun fonce froid", "Chatain fonce cendre"],
            "avoid": ["Roux", "Auburn", "Dore", "Caramel", "Blond chaud"]
        },
        "accessories": {
            "glasses": "Noir, argent, bleu marine, transparents froids",
            "jewelry": "Argent, platine, diamant, saphir bleu, amethyste, tanzanite",
            "bags_shoes": "Noir, marine, gris charbon, bordeaux froid."
        }
    },

    "Bright Winter": {
        "description": (
            "Eclat glacial. Les couleurs les plus vives et pures sur fond froid. "
            "Bleu electrique, fuchsia vif, jaune citron, vert emeraude — eclat maximal."
        ),
        "palette_neutrals": ["#F8F8FF", "#B0B0B0", "#4A4A4A", "#1C1C1C", "#000000"],
        "palette_accents": ["#0066FF", "#FF1493", "#FFF700", "#8B00FF", "#FF0000", "#00CC44"],
        "palette_avoid": ["#C2B280", "#8B8C7A", "#A39BA8", "#CEB5A7", "#7F6B5D", "#B5838D"],
        "metals": "Argent brillant, or blanc, chrome, platine.",
        "makeup": {
            "foundation": "Sous-ton neutre a froid. Finish lumineux ou satin.",
            "lips": ["Fuchsia vif", "Rouge pur", "Rose electrique", "Baie vive"],
            "eyes": ["Bleu electrique", "Violet vif", "Argent", "Vert emeraude", "Noir net"],
            "blush": ["Rose vif", "Fuchsia", "Berry eclatant"],
            "eyebrows": "Brun fonce net, noir — trait precis"
        },
        "clothing": {
            "best_combinations": [
                "Bleu electrique + blanc",
                "Fuchsia + noir",
                "Emeraude + argent",
                "Jaune citron + marine"
            ],
            "patterns": "Color-block audacieux, geometrique, minimaliste, pop art",
            "contrast_tip": "Contraste MAXIMAL. Couleurs pures et saturees, contrastes nets.",
            "fabrics": "Soie brillante, cuir patent, vinyle, satin, materiaux a reflets, sequins"
        },
        "hair": {
            "ideal": ["Noir brillant", "Brun fonce froid", "Noir bleu"],
            "avoid": ["Cendre terne", "Dore", "Roux", "Chatain moyen"]
        },
        "accessories": {
            "glasses": "Noir net, couleurs vives, formes geometriques",
            "jewelry": "Argent brillant, diamant, cristal, pierres vives (saphir, emeraude, rubis)",
            "bags_shoes": "Noir, couleurs vives pures, metallise. Le terne est interdit."
        }
    },

    "True Winter": {
        "description": (
            "L'essence de l'hiver. Equilibre parfait entre froid, profondeur et nettete. "
            "Rouge vrai, bleu pur, noir, blanc — clarte absolue."
        ),
        "palette_neutrals": ["#F5F5F5", "#C0C0C0", "#36454F", "#1C1C1C", "#000000"],
        "palette_accents": ["#CC0000", "#0000CD", "#00563F", "#FF69B4", "#301934", "#4169E1"],
        "palette_avoid": ["#DDA15E", "#C2B280", "#F4A261", "#8B8C7A", "#D2B48C", "#FFE5B4"],
        "metals": "Argent, platine, or blanc.",
        "makeup": {
            "foundation": "Sous-ton neutre froid equilibre. Net et impeccable.",
            "lips": ["Rouge classique", "Rose vif froid", "Prune", "Berry"],
            "eyes": ["Gris charbon", "Bleu profond", "Argent", "Vert fonce froid", "Prune"],
            "blush": ["Rose froid vif", "Berry", "Prune douce"],
            "eyebrows": "Brun fonce a noir, bien definis"
        },
        "clothing": {
            "best_combinations": [
                "Noir + blanc pur",
                "Marine + rouge",
                "Gris + fuchsia",
                "Emeraude froid + noir"
            ],
            "patterns": "Classiques nets, rayures, geometrique, monochrome",
            "contrast_tip": "Contraste eleve. Purete et nettete, le noir et blanc est votre signature.",
            "fabrics": "Laine fine, soie, coton net, cuir lisse, cachemire, gabardine"
        },
        "hair": {
            "ideal": ["Noir profond", "Brun fonce espresso", "Chatain fonce froid"],
            "avoid": ["Roux", "Dore", "Caramel", "Cuivre"]
        },
        "accessories": {
            "glasses": "Noir, argent, formes classiques nettes",
            "jewelry": "Argent, platine, diamant, perles blanches, onyx, saphir",
            "bags_shoes": "Noir, marine, blanc, gris. Classique et impeccable."
        }
    },
}


# Required keys for data integrity validation
REQUIRED_ADVICE_KEYS = [
    "description", "palette_neutrals", "palette_accents", "palette_avoid",
    "metals", "makeup", "clothing", "hair", "accessories",
]
REQUIRED_MAKEUP_KEYS = ["foundation", "lips", "eyes", "blush", "eyebrows"]
REQUIRED_CLOTHING_KEYS = ["best_combinations", "patterns", "contrast_tip", "fabrics"]
REQUIRED_HAIR_KEYS = ["ideal", "avoid"]
REQUIRED_ACCESSORIES_KEYS = ["glasses", "jewelry", "bags_shoes"]
