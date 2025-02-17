import { db } from './firebase';
import { doc, setDoc, getDoc, collection, query, where, getDocs, updateDoc, increment, Timestamp, orderBy } from 'firebase/firestore';

interface UserData {
  uid: string;
  email: string;
  name: string;
  username: string;
  remainingFreeDebates: number;
  totalDebates: number;
  wins: number;
  losses: number;
  draws: number;
}

interface Debate {
  id: string;
  topic: string;
  opponent: string;
  date: string;
  result: 'win' | 'loss' | 'draw' | 'ongoing';
}

export const createUserDocument = async (userData: Partial<UserData>): Promise<void> => {
  const userRef = doc(db, 'users', userData.uid!);
  await setDoc(userRef, {
    ...userData,
    remainingFreeDebates: 45,  // Updated to match the initial value in your database
    totalDebates: 0,
    wins: 0,
    losses: 0,
    draws: 0,
  }, { merge: true });
};

export const isUsernameAvailable = async (username: string): Promise<boolean> => {
  const usersRef = collection(db, 'users');
  const q = query(usersRef, where('username', '==', username));
  const querySnapshot = await getDocs(q);
  return querySnapshot.empty;
};

export const getUserDocument = async (uid: string): Promise<UserData | null> => {
  const userRef = doc(db, 'users', uid);
  const userSnap = await getDoc(userRef);
  
  if (userSnap.exists()) {
    return userSnap.data() as UserData;
  } else {
    return null;
  }
};

export const updateUserDocument = async (uid: string, data: Partial<UserData>): Promise<void> => {
  const userRef = doc(db, 'users', uid);
  await updateDoc(userRef, data);
};

export const getUserDebates = async (uid: string): Promise<Debate[]> => {
  const debatesRef = collection(db, 'debates');
  const q = query(debatesRef, where('participants', 'array-contains', uid));
  const querySnapshot = await getDocs(q);
  
  const debates: Debate[] = [];
  querySnapshot.forEach((doc) => {
    const data = doc.data();
    debates.push({
      id: doc.id,
      topic: data.topic,
      opponent: data.participants.find((p: any) => p.userId !== uid)?.displayName || 'Unknown',
      date: data.createdAt.toDate().toLocaleDateString(),
      result: data.status === 'ongoing' ? 'ongoing' : (data.winner === uid ? 'win' : (data.loser === uid ? 'loss' : 'draw')),
    });
  });
  
  return debates;
};

export const getUserStats = async (uid: string): Promise<Partial<UserData>> => {
  const userRef = doc(db, 'users', uid);
  const userSnap = await getDoc(userRef);
  
  if (userSnap.exists()) {
    const userData = userSnap.data() as UserData;
    console.log("Raw user data from Firestore:", userData);
    return {
      totalDebates: userData.totalDebates || 0,
      wins: userData.wins || 0,
      losses: userData.losses || 0, 
      draws: userData.draws || 0,
      remainingFreeDebates: userData.remainingFreeDebates || 45,
    };
  } else {
    throw new Error('User not found');
  }
};

export const createDebate = async (topic: string, userId: string, opponentId: string): Promise<string> => {
  const debateRef = doc(collection(db, 'debates'));
  const userSnap = await getDoc(doc(db, 'users', userId));
  const opponentSnap = await getDoc(doc(db, 'users', opponentId));
  
  if (!userSnap.exists() || !opponentSnap.exists()) {
    throw new Error('User or opponent not found');
  }
  
  const userData = userSnap.data();
  const opponentData = opponentSnap.data();
  
  await setDoc(debateRef, {
    topic,
    createdAt: Timestamp.now(),
    status: 'ongoing',
    participants: [
      {
        userId: userId,
        displayName: userData.name,
      },
      {
        userId: opponentId,
        displayName: opponentData.name,
      },
    ],
    winner: null,
    loser: null,
  });
  
  // Decrement remaining free debates for both users
  await updateDoc(doc(db, 'users', userId), {
    remainingFreeDebates: increment(-1),
    totalDebates: increment(1),
  });
  
  await updateDoc(doc(db, 'users', opponentId), {
    remainingFreeDebates: increment(-1),
    totalDebates: increment(1),
  });
  
  return debateRef.id;
};

export const addDebateMessage = async (debateId: string, userId: string, content: string, isAIAssisted: boolean): Promise<void> => {
  const messageRef = doc(collection(db, `debateMessages/${debateId}/messages`));
  await setDoc(messageRef, {
    userId,
    content,
    timestamp: Timestamp.now(),
    isAIAssisted,
  });
};

export const getDebateMessages = async (debateId: string): Promise<any[]> => {
  const messagesRef = collection(db, `debateMessages/${debateId}/messages`);
  const q = query(messagesRef, orderBy('timestamp'));
  const querySnapshot = await getDocs(q);
  
  const messages: any[] = [];
  querySnapshot.forEach((doc) => {
    messages.push({ id: doc.id, ...doc.data() });
  });
  
  return messages;
};

export const completeDebate = async (debateId: string, winnerId: string, loserId: string): Promise<void> => {
  const debateRef = doc(db, 'debates', debateId);
  await updateDoc(debateRef, {
    status: 'completed',
    winner: winnerId,
    loser: loserId,
  });
  
  // Update user stats
  await updateDoc(doc(db, 'users', winnerId), {
    wins: increment(1),
  });
  
  await updateDoc(doc(db, 'users', loserId), {
    losses: increment(1),
  });
};