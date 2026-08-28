import { useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControlsLite } from "./OrbitControlsLite.jsx";

// Section 25A spec item 36: instanced mesh, not N independent objects --
// this scene must stay smooth even at 1024 threads (32x32, the largest
// block preset this app offers).
function ThreadCubes({ blockX, blockY, warpSize, showWarps, hoveredId, onHover, onLeave, onSelect }) {
  const meshRef = useRef();
  const count = blockX * blockY;
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    for (let i = 0; i < count; i++) {
      const tx = i % blockX;
      const ty = Math.floor(i / blockX);
      dummy.position.set((tx - (blockX - 1) / 2) * 0.55, (ty - (blockY - 1) / 2) * -0.55, 0);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }, [blockX, blockY, count, dummy]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const color = new THREE.Color();
    for (let i = 0; i < count; i++) {
      if (i === hoveredId) {
        color.set("#ffffff");
      } else if (showWarps) {
        const warpIdx = Math.floor(i / warpSize);
        color.setHSL((warpIdx * 0.17) % 1, 0.65, 0.55);
      } else {
        color.set("#2fe6a8");
      }
      mesh.setColorAt(i, color);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [hoveredId, showWarps, count, warpSize]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[null, null, count]}
      onPointerMove={(e) => { e.stopPropagation(); if (e.instanceId != null) onHover(e.instanceId); }}
      onPointerOut={(e) => { e.stopPropagation(); onLeave(); }}
      onClick={(e) => { e.stopPropagation(); if (e.instanceId != null) onSelect(e.instanceId); }}
    >
      <boxGeometry args={[0.46, 0.46, 0.46]} />
      <meshStandardMaterial roughness={0.35} metalness={0.2} />
    </instancedMesh>
  );
}

export function ThreadBlockScene({ blockX, blockY, warpSize, showWarps, hoveredId, onHover, onLeave, onSelect, autoRotate, presetRef }) {
  return (
    <Canvas camera={{ position: [0, 0, Math.max(blockX, blockY) * 0.6 + 3], fov: 45 }} dpr={[1, 1.5]}>
      <color attach="background" args={["#0a0e18"]} />
      <ambientLight intensity={0.6} />
      <pointLight position={[4, 4, 6]} intensity={50} color="#5b8cff" />
      <pointLight position={[-4, -4, 4]} intensity={25} color="#2fe6a8" />
      <ThreadCubes blockX={blockX} blockY={blockY} warpSize={warpSize} showWarps={showWarps} hoveredId={hoveredId} onHover={onHover} onLeave={onLeave} onSelect={onSelect} />
      <OrbitControlsLite presetRef={presetRef} autoRotate={autoRotate} />
    </Canvas>
  );
}
