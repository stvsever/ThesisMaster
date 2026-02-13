import json
import os
from collections import defaultdict
from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD, DC, DCTERMS

# Define base namespace
BASE_IRI = "http://mental_health_disorders.org/ontology#"  # Change to desired base IRI
BASE = Namespace(BASE_IRI)


def sanitize_label(label):
    """
    Convert a label to a valid IRI fragment by replacing spaces and special characters.
    """
    sanitized = label.strip().replace(" ", "_").replace("-", "_").lower()
    sanitized = "".join(char for char in sanitized if char.isalnum() or char == "_")
    return sanitized


def add_class(g, label, parent_labels=None, path=None, class_depths=None):
    """
    Add a class with the given label to the graph, tracking depth and path.
    """
    if parent_labels is None:
        parent_labels = []
    if path is None:
        path = []
    if class_depths is None:
        class_depths = {}

    current_path = path + [label]
    class_fragment = sanitize_label("_".join(current_path))
    class_iri = BASE[class_fragment]

    # Register class depth before adding to graph
    depth = len(current_path)
    class_depths[class_iri] = depth

    # Class declaration
    g.add((class_iri, RDF.type, OWL.Class))
    g.add((class_iri, RDFS.label, Literal(label, datatype=XSD.string)))

    # Parent relationship
    if parent_labels:
        parent_fragment = sanitize_label("_".join(parent_labels))
        parent_iri = BASE[parent_fragment]
        g.add((class_iri, RDFS.subClassOf, parent_iri))

    return class_iri, current_path


def _add_criteria_list(g, criteria_list, parent_path, class_depths):
    """
    Create child classes for each criterion string in a list under the given parent path.
    """
    if not isinstance(criteria_list, list):
        return
    for crit in criteria_list:
        if isinstance(crit, str) and crit.strip():
            add_class(
                g,
                crit,
                parent_labels=parent_path,
                path=parent_path,
                class_depths=class_depths
            )


def _process_units_of_analysis(g, units_mapping, parent_path, class_depths):
    """
    Handle 'units_of_analysis' blocks of the form:
    { "genes": [...], "molecules": [...], ... }
    Each unit name becomes a class under the parent, and each item in the list
    becomes a class under that unit.
    """
    if not isinstance(units_mapping, dict):
        return
    for unit_name, criteria in units_mapping.items():
        # Add the unit of analysis as a class under the current node
        _, unit_path = add_class(
            g,
            unit_name,
            parent_labels=parent_path,
            path=parent_path,
            class_depths=class_depths
        )
        # Add criteria (list or nested)
        if isinstance(criteria, list):
            _add_criteria_list(g, criteria, unit_path, class_depths)
        elif isinstance(criteria, dict):
            # If some units themselves are dicts (rare), recurse in a generic way
            process_hierarchy(
                g,
                {unit_name: criteria},
                parent_labels=parent_path,
                path=parent_path,
                class_depths=class_depths
            )


def process_hierarchy(g, hierarchy, parent_labels=None, path=None, class_depths=None):
    """
    Recursively process hierarchy with depth tracking.

    Supports:
    - Arbitrary nested dicts of named nodes.
    - 'units_of_analysis' blocks mapping unit names -> list of criteria.
    - Bare lists at any node treated as criteria under that node.
    - Ignores metadata fields like prevalence.

    Semantics:
    Each (key -> value) pair:
      - key becomes a class.
      - if value is dict: recurse into its keys, but specially handle 'units_of_analysis'.
      - if value is list: each string item becomes a child class (criterion).
    """
    if parent_labels is None:
        parent_labels = []
    if path is None:
        path = []
    if class_depths is None:
        class_depths = {}

    METADATA_KEYS = {
        "estimated_prevalence_percent",
        "prevalence",
        "prevalence_percent",
        "notes",
        "description",
    }

    for label, node in hierarchy.items():
        _, current_path = add_class(
            g, label, parent_labels, path, class_depths
        )

        # Dict node: handle units_of_analysis + recurse into other keys
        if isinstance(node, dict):
            # 1) Units of analysis (primary requirement)
            if "units_of_analysis" in node:
                _process_units_of_analysis(
                    g, node["units_of_analysis"], current_path, class_depths
                )

            # 2) Criteria lists possibly stored directly under the node
            if "criteria" in node and isinstance(node["criteria"], list):
                _add_criteria_list(g, node["criteria"], current_path, class_depths)

            # 3) Recurse into remaining keys
            for k, v in node.items():
                if k in METADATA_KEYS or k in {"units_of_analysis", "criteria"}:
                    continue
                if isinstance(v, dict):
                    process_hierarchy(
                        g,
                        {k: v},
                        parent_labels=current_path,
                        path=current_path,
                        class_depths=class_depths
                    )
                elif isinstance(v, list):
                    # A bare list under a named child -> create that child then add items as criteria
                    # Make the child a class (k), then add list items underneath
                    _, child_path = add_class(
                        g, k, parent_labels=current_path, path=current_path, class_depths=class_depths
                    )
                    _add_criteria_list(g, v, child_path, class_depths)

        # List node: treat as criteria under the current label
        elif isinstance(node, list):
            _add_criteria_list(g, node, current_path, class_depths)

        # Other scalar types are ignored (if any)


def load_json_hierarchy(json_path):
    """
    Load and validate JSON hierarchy with basic error handling.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON root should be a dictionary")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load {json_path}") from e


def add_ontology_metadata(g, title, iri, description=None, creator=None, version=None):
    """
    Add comprehensive ontology metadata using DC and DCTERMS.
    """
    ontology = URIRef(iri)
    g.add((ontology, RDF.type, OWL.Ontology))
    g.add((ontology, DC.title, Literal(title, datatype=XSD.string)))

    if description:
        g.add((ontology, DC.description, Literal(description, datatype=XSD.string)))
    if creator:
        g.add((ontology, DC.creator, Literal(creator, datatype=XSD.string)))
    if version:
        g.add((ontology, OWL.versionInfo, Literal(version, datatype=XSD.string)))


def describe_ontology(g, class_depths):
    """Generate comprehensive ontology analysis report."""
    # Initialize summary data
    ontology_node = next(g.subjects(RDF.type, OWL.Ontology), None)
    summary = {
        "title": str(g.value(ontology_node, DC.title)) if ontology_node else "Unnamed PHOENIX_ontology",
        "version": str(g.value(ontology_node, OWL.versionInfo)) if ontology_node else "Unversioned",
        "class_counts": {
            "total": 0,
            "roots": 0,
            "leaves": 0
        },
        "depth_analysis": {
            "max": 0,
            "min": 0,
            "avg": 0,
            "distribution": defaultdict(int)
        },
        "duplicates": {
            "total_labels": 0,
            "multi_depth": 0,
            "exact_duplicates": 0
        }
    }

    # Get all classes
    classes = list(g.subjects(RDF.type, OWL.Class))
    summary["class_counts"]["total"] = len(classes)

    # Calculate depth statistics
    depths = list(class_depths.values())
    if depths:
        summary["depth_analysis"]["max"] = max(depths)
        summary["depth_analysis"]["min"] = min(depths)
        summary["depth_analysis"]["avg"] = sum(depths) / len(depths)
        for d in depths:
            summary["depth_analysis"]["distribution"][d] += 1

    # Calculate root and leaf classes
    summary["class_counts"]["roots"] = len([c for c in classes if not list(g.objects(c, RDFS.subClassOf))])
    summary["class_counts"]["leaves"] = len([c for c in classes if not list(g.subjects(RDFS.subClassOf, c))])

    # Duplicate analysis
    label_map = defaultdict(list)
    depth_map = defaultdict(set)
    for cls in classes:
        label = str(g.value(cls, RDFS.label))
        label_map[label].append(cls)
        depth_map[label].add(class_depths[cls])

    # Calculate duplicate metrics
    exact_duplicates = 0
    multi_depth_duplicates = 0
    for label, items in label_map.items():
        if len(items) > 1:
            exact_duplicates += 1
        if len(depth_map[label]) > 1:
            multi_depth_duplicates += 1

    summary["duplicates"]["total_labels"] = exact_duplicates
    summary["duplicates"]["multi_depth"] = multi_depth_duplicates

    # Print report
    print("\n=== PHOENIX_ontology Analysis Report ===")
    print(f"Title: {summary['title']}")
    print(f"Version: {summary['version']}")

    print("\nClass Hierarchy:")
    print(f"Total classes: {summary['class_counts']['total']}")
    print(f"Root classes: {summary['class_counts']['roots']}")
    print(f"Leaf classes: {summary['class_counts']['leaves']}")

    print("\nDepth Analysis:")
    print(f"Maximum depth: {summary['depth_analysis']['max']}")
    print(f"Minimum depth: {summary['depth_analysis']['min']}")
    print(f"Average depth: {summary['depth_analysis']['avg']:.1f}")
    print("\nDepth Distribution:")
    for depth in sorted(summary["depth_analysis"]["distribution"]):
        count = summary["depth_analysis"]["distribution"][depth]
        print(f"  Depth {depth}: {count} classes")

    print("\nDuplicate Analysis:")
    print(f"Total label duplicates: {summary['duplicates']['total_labels']}")
    print(f"Labels appearing at multiple depths: {summary['duplicates']['multi_depth']}")

    return summary


def main():
    # Configuration
    input_json_path = "/SystemComponents/PHOENIX_ontology/separate/CRITERION/steps/01_raw/clinical/02_post_generation/RDoC_pg/RDoC_criteria_merged.json"  # Single json file
    output_dir = "/SystemComponents/PHOENIX_ontology/separate/CRITERION/steps/01_raw/clinical/02_post_generation/RDoC_pg"  # Output directory
    output_name = os.path.splitext(os.path.basename(input_json_path))[0] + ".owl"  # Name based on input JSON
    output_path = os.path.join(output_dir, output_name)

    # Initialize graph and namespaces
    g = Graph()
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("dc", DC)
    g.bind("dcterms", DCTERMS)
    g.bind("base", BASE)

    # Add ontology metadata
    add_ontology_metadata(
        g,
        title="PHOENIX PHOENIX_ontology - subset: RDoC-based ontology with units of analysis and generated criteria",
        iri=BASE_IRI,
        creator="Stijn Van Severen",
        version="1.0"
    )

    # Process JSON file
    class_depths = {}
    processed_files = 0

    try:
        hierarchy = load_json_hierarchy(input_json_path)
        process_hierarchy(g, hierarchy, class_depths=class_depths)
        processed_files += 1
        print(f"Processed: {os.path.basename(input_json_path)}")
    except Exception as e:
        print(f"Error processing {os.path.basename(input_json_path)}: {str(e)}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save and report
    g.serialize(destination=output_path, format="xml")
    print(f"\nPHOENIX_ontology saved to: {output_path}")
    print(f"Processed {processed_files} JSON files")

    # Generate detailed analysis
    return describe_ontology(g, class_depths)


if __name__ == "__main__":
    main()
