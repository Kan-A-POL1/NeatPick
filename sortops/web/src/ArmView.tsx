import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import URDFLoader, { type URDFRobot } from "urdf-loader";

const URDF_PATH = "/urdf/so101_follower.urdf";
/** Normalized units between target and measured joints below which the target copy is hidden. */
const GHOST_MIN_GAP = 2.5;

/** Normalized SO-101 position → URDF angle: the mapping Physical AI Studio's own 3D view uses. */
function applyJoints(robot: URDFRobot, names: string[], values: number[]): void {
  names.forEach((name, index) => {
    const joint = robot.joints[name];
    const value = values[index];
    if (!joint || value === undefined) return;
    const fraction = name === "gripper" ? value / 100 : (value + 100) / 200;
    const lower = Number(joint.limit.lower);
    const upper = Number(joint.limit.upper);
    joint.setJointValue(lower + (upper - lower) * Math.min(1, Math.max(0, fraction)));
  });
}

function paint(robot: URDFRobot, material: THREE.Material): void {
  robot.traverse((child) => {
    if (child instanceof THREE.Mesh) child.material = material;
  });
}

interface Props {
  names: string[] | undefined;
  joints: number[] | undefined;
  ghost: number[] | null | undefined;
  showGhost: boolean;
  inspectionPose: number[] | undefined;
}

export function ArmView({ names, joints, ghost, showGhost, inspectionPose }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const real = useRef<URDFRobot | null>(null);
  const shadow = useRef<URDFRobot | null>(null);
  const marker = useRef<THREE.Mesh | null>(null);
  const failed = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = host.current;
    if (!container) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 10);
    camera.position.set(0.55, 0.42, 0.55);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0.12, 0.12, 0);
    controls.enableDamping = true;

    scene.add(new THREE.HemisphereLight(0xdde6ff, 0x1a1d24, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(1, 2, 1.5);
    scene.add(key);

    const grid = new THREE.GridHelper(1, 20, 0x3a4252, 0x262c37);
    scene.add(grid);

    // URDF is z-up; three.js is y-up.
    const world = new THREE.Group();
    world.rotation.x = -Math.PI / 2;
    scene.add(world);

    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 24, 16),
      new THREE.MeshBasicMaterial({ color: 0x5fb4ff, transparent: true, opacity: 0.55 }),
    );
    world.add(sphere);
    marker.current = sphere;

    const solid = new THREE.MeshStandardMaterial({ color: 0xd9dde6, metalness: 0.1, roughness: 0.55 });
    const ghostMaterial = new THREE.MeshBasicMaterial({
      color: 0xffc35a,
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
    });

    let disposed = false;
    // Load the URDF once; the policy-target copy is a clone made after every mesh arrived (manager.onLoad).
    // Two parallel loads share three.js's in-flight requests and the second manager never sees its meshes.
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    let loaded: URDFRobot | null = null;
    manager.onLoad = () => {
      if (disposed || !loaded) return;
      paint(loaded, solid);
      real.current = loaded;
      const copy = loaded.clone(true) as URDFRobot;
      paint(copy, ghostMaterial);
      copy.visible = false;
      world.add(copy);
      shadow.current = copy;
    };
    manager.onError = (url) => console.warn(`URDF asset failed: ${url}`);
    loader.load(
      URDF_PATH,
      (robot) => {
        loaded = robot;
        if (!disposed) world.add(robot);
      },
      undefined,
      () => {
        if (failed.current) failed.current.hidden = false;
      },
    );

    const resize = () => {
      const { clientWidth, clientHeight } = container;
      renderer.setSize(clientWidth, clientHeight);
      camera.aspect = clientWidth / Math.max(1, clientHeight);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    let frame = 0;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    if (real.current && names && joints) applyJoints(real.current, names, joints);
  }, [names, joints]);

  useEffect(() => {
    const robot = shadow.current;
    if (!robot) return;
    // Only draw the target where it says something: when it leads the measured arm by a visible margin.
    const apart = !!ghost && !!joints && ghost.some((value, index) => Math.abs(value - (joints[index] ?? value)) > GHOST_MIN_GAP);
    robot.visible = showGhost && apart;
    if (names && ghost) applyJoints(robot, names, ghost);
  }, [names, ghost, joints, showGhost]);

  useEffect(() => {
    const [x, y, z] = inspectionPose ?? [];
    if (marker.current && x !== undefined && y !== undefined && z !== undefined) marker.current.position.set(x, y, z);
  }, [inspectionPose]);

  return (
    <div className="arm-view" ref={host}>
      <div ref={failed} className="cam-empty" hidden>
        3D model unavailable (URDF not served)
      </div>
    </div>
  );
}
