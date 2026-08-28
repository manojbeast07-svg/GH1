import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControlsLite } from "./OrbitControlsLite.jsx";

function SmTile({ position, index, hovered, onHover, onLeave, onSelect }) {
  return (
    <mesh
      position={position}
      onPointerOver={(e) => { e.stopPropagation(); onHover(index); }}
      onPointerOut={(e) => { e.stopPropagation(); onLeave(index); }}
      onClick={(e) => { e.stopPropagation(); onSelect(index); }}
    >
      <boxGeometry args={[0.82, 0.82, 0.32]} />
      <meshStandardMaterial
        color={hovered ? "#2fe6a8" : "#20283d"}
        emissive={hovered ? "#2fe6a8" : "#000000"}
        emissiveIntensity={hovered ? 0.7 : 0}
        roughness={0.4}
        metalness={0.3}
      />
    </mesh>
  );
}

function Die({ count, hoveredIndex, onHover, onLeave, onSelect, spin, presetRef }) {
  const groupRef = useRef();
  const cols = Math.max(1, Math.ceil(Math.sqrt(count * 1.6)));
  const rows = Math.max(1, Math.ceil(count / cols));

  const positions = useMemo(() => {
    const arr = [];
    for (let i = 0; i < count; i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      arr.push([(c - (cols - 1) / 2) * 1.0, (r - (rows - 1) / 2) * -1.0, 0]);
    }
    return arr;
  }, [count, cols, rows]);

  useFrame((_, delta) => {
    if (spin && groupRef.current) groupRef.current.rotation.z += delta * 0.05;
  });

  return (
    <group ref={groupRef}>
      <mesh position={[0, 0, -0.35]}>
        <boxGeometry args={[cols * 1.05 + 0.4, rows * 1.05 + 0.4, 0.25]} />
        <meshStandardMaterial color="#080b12" roughness={0.7} />
      </mesh>
      {positions.map((p, i) => (
        <SmTile key={i} position={p} index={i} hovered={hoveredIndex === i} onHover={onHover} onLeave={onLeave} onSelect={onSelect} />
      ))}
    </group>
  );
}

export function GpuChipScene({ smCount, hoveredIndex, onHover, onLeave, onSelect, autoRotate, presetRef }) {
  return (
    <Canvas camera={{ position: [4, 3, 6], fov: 42 }} dpr={[1, 1.5]} gl={{ antialias: true }}>
      <color attach="background" args={["#0a0e18"]} />
      <ambientLight intensity={0.55} />
      <pointLight position={[5, 5, 6]} intensity={60} color="#5b8cff" />
      <pointLight position={[-5, -3, 4]} intensity={30} color="#2fe6a8" />
      <Die count={smCount} hoveredIndex={hoveredIndex} onHover={onHover} onLeave={onLeave} onSelect={onSelect} spin={autoRotate} presetRef={presetRef} />
      <OrbitControlsLite presetRef={presetRef} autoRotate={autoRotate} />
    </Canvas>
  );
}
