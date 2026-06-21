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
  /**
   * Normalise an arbitrary string the same way the JSON keys were built:
   * lowercase, strip non-alphanumeric characters (spaces, hyphens, underscores, punctuation).
   * e.g. "Fruit Fly" → "fruitfly",  "tiger_mosquito" → "tigermosquito"
   */
  const normalise = (name) =>
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "");

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
          // keep original key for debugging/compatibility
          map[r.key] = r;
          // add normalized variants so lookups work for names, keys, and variants
          try {
            if (r.key) map[normalise(r.key)] = r;
            if (r.name) map[normalise(r.name)] = r;
          } catch (e) {
            console.error(`Error normalising insect record: ${r.key || r.name}`, e);
          }
        });
        setLookup(map);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  /**
   * Returns the matching insect record or null if not found / data not loaded.
   */
  const lookupInsect = (name) => {
    if (!lookup || !name) return null;
    return lookup[normalise(name)] ?? null;
  };

  return { lookupInsect, loading, error };
}