// Offscreen character thumbnails.
//
// Rendered once per character and stored in the library, so the picker never
// has to re-render on open. The look is decided per asset: a character whose
// own textures actually resolved is shown with its authored materials (MMD
// toon ramps, sphere maps, FBX materials); one with missing or failed textures
// falls back to the white look so a broken bake never becomes the thumbnail.
import * as THREE from "three";

const THUMBNAIL_SIZE = 256;
const CAMERA_FOV = 30;
const FRAMING_FILL = 0.78;
const VIEW_DIRECTION = new THREE.Vector3(0.38, 0.1, 1).normalize();

function hasUsableTextures(report) {
  if (!report) return false;
  const loaded = Number(report.loaded) || 0;
  const failed = Number(report.failed) || 0;
  return loaded > 0 && failed === 0;
}

/** The look a thumbnail would use, so the UI can label it before rendering. */
export function thumbnailLook(asset) {
  return hasUsableTextures(asset?.textureReport) ? "original" : "white";
}

function frameCamera(camera, root) {
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) return false;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.5;
  if (!Number.isFinite(radius) || radius <= 0) return false;
  const halfFov = (camera.fov * Math.PI) / 360;
  const distance = (radius / Math.sin(halfFov)) / FRAMING_FILL;
  camera.position.copy(center).addScaledVector(VIEW_DIRECTION, distance);
  camera.lookAt(center.x, center.y - size.y * 0.04, center.z);
  camera.near = Math.max(0.001, distance - radius * 4);
  camera.far = distance + radius * 6;
  camera.updateProjectionMatrix();
  return true;
}

/**
 * Render one character to a data URL.
 * Resolves to `{ dataUrl, look, size }`.
 */
// One WebGL context for every thumbnail. Creating and tearing down a context
// per character costs far more than the render itself (context creation, MMD
// toon shader compile, texture upload), and a bulk import would pay that once
// per model. The context is kept alive and only the scene contents are swapped.
let thumbRenderer = null;
let thumbScene = null;
let thumbCamera = null;
function thumbnailRenderer(size) {
  if (!thumbRenderer) {
    thumbRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, preserveDrawingBuffer: true });
    thumbRenderer.setPixelRatio(1);
    thumbRenderer.outputColorSpace = THREE.SRGBColorSpace;
    thumbRenderer.toneMapping = THREE.NoToneMapping;
    thumbRenderer.shadowMap.enabled = false;
    thumbRenderer.setClearColor(0x000000, 0);
    thumbScene = new THREE.Scene();
    const key = new THREE.DirectionalLight(0xffffff, 2.0);
    key.position.set(1.4, 2.6, 2.8);
    const fill = new THREE.DirectionalLight(0xe4ecff, 0.9);
    fill.position.set(-2.4, 1.2, 1.8);
    const rim = new THREE.DirectionalLight(0xffffff, 1.1);
    rim.position.set(-0.9, 1.8, -2.6);
    thumbScene.add(key, fill, rim, new THREE.AmbientLight(0xffffff, 0.6));
    thumbCamera = new THREE.PerspectiveCamera(CAMERA_FOV, 1, 0.01, 1000);
  }
  if (thumbRenderer.domElement.width !== size) thumbRenderer.setSize(size, size, false);
  return { renderer: thumbRenderer, scene: thumbScene, camera: thumbCamera };
}

export async function renderCharacterThumbnail(asset, { size = THUMBNAIL_SIZE } = {}) {
  const look = thumbnailLook(asset);
  const restore = [];
  let root = null;
  let ownsRoot = false;
  let override = null;
  try {
    if (asset?.mesh) {
      root = asset.mesh;
    } else if (typeof asset?.clone === "function") {
      root = asset.clone();
      ownsRoot = true;
    }
    if (!root) throw new Error("Character has no renderable mesh");
    if (!root.parent && !root.isMesh) root.updateMatrixWorld(true);

    if (look === "white") {
      override = new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.92, metalness: 0 });
      root.traverse((node) => {
        if (!node.isMesh || !node.material) return;
        restore.push([node, node.material]);
        node.material = override;
      });
    }

    const { renderer, scene, camera } = thumbnailRenderer(size);

    scene.add(root);
    root.updateMatrixWorld(true);
    if (!frameCamera(camera, root)) throw new Error("Character bounds are empty");

    renderer.render(scene, camera);
    const dataUrl = renderer.domElement.toDataURL("image/png");
    if (!dataUrl.startsWith("data:image/png;base64,")) throw new Error("Thumbnail encode failed");
    return { dataUrl, look, size };
  } finally {
    // Never leave the shared template wearing the white look.
    for (const [node, material] of restore) node.material = material;
    if (root) {
      root.removeFromParent();
      if (ownsRoot && typeof asset?.releaseClone === "function") asset.releaseClone(root);
    }
    override?.dispose();
  }
}
