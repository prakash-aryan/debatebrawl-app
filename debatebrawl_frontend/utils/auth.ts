import { auth } from './firebase';
import { 
  GoogleAuthProvider,
  signInWithPopup,
  User,
} from "firebase/auth";

export const signInWithGoogle = async (): Promise<User> => {
  const provider = new GoogleAuthProvider();
  const userCredential = await signInWithPopup(auth, provider);
  return userCredential.user;
};

export const logOut = async (): Promise<void> => {
  await auth.signOut();
};

export const getCurrentUser = (): User | null => {
  return auth.currentUser;
};

export const isEmailVerified = (user: User | null): boolean => {
  return user?.emailVerified || false;
};