import { useState, useEffect } from "react";

/**
 * Loads /public/insects.json once and exposes a lookup function.
 *
 * Usage:
 *   const { lookupInsect, loading, error } = useInsectData();
 *   const info = lookupInsect("Fruit Fly");  // → insect object or null
 */
export function useInsectData() {
  const [lookup, setLookup] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/insects.json")
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load knowledge base (${res.status})`);
        return res.json();
      })
      .then((records) => {
        // Build a map keyed by the normalised name for fast O(1) lookup
        const map = {};
        records.forEach((r) => {
          map[r.key] = r;
        });
        setLookup(map);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  /**
   * Normalise an arbitrary string the same way the JSON keys were built:
   * lowercase, strip spaces and hyphens.
   * e.g. "Fruit Fly" → "fruitfly",  "Bed-bug" → "bedbug"
   */
  const normalise = (name) =>
    name
      .toLowerCase()
      .replace(/[\s\-]+/g, "");

  /**
   * Returns the matching insect record or null if not found / data not loaded.
   */
  const lookupInsect = (name) => {
    if (!lookup || !name) return null;
    return lookup[normalise(name)] ?? null;
  };

  return { lookupInsect, loading, error };
}